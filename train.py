"""Trainining script for WaveNet vocoder

usage: train.py [options]

options:
    --dump-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt

import sys

import os
from os.path import dirname, join, expanduser, exists
from tqdm import tqdm
from datetime import datetime
import random
import json
from glob import glob

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lrschedule

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler

from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

import librosa.display

from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn

from wavenet_vocoder import WaveNet
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
from wavenet_vocoder.mixture import discretized_mix_logistic_loss
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
from wavenet_vocoder.mixture import mix_gaussian_loss
from wavenet_vocoder.mixture import sample_from_mix_gaussian

import audio
from hparams import hparams, hparams_debug_string

global_step = 0
global_test_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = True


def sanity_check(model, c, g):
    if model.has_speaker_embedding():
        if g is None:
            raise RuntimeError(
                "WaveNet expects speaker embedding, but speaker-id is not provided")
    else:
        if g is not None:
            raise RuntimeError(
                "WaveNet expects no speaker embedding, but speaker-id is provided")

    if model.local_conditioning_enabled():
        if c is None:
            raise RuntimeError("WaveNet expects conditional features, but not given")
    else:
        if c is not None:
            raise RuntimeError("WaveNet expects no conditional features, but given")


def maybe_set_epochs_based_on_max_steps(hp, steps_per_epoch):
    nepochs = hp.nepochs
    max_train_steps = hp.max_train_steps
    if max_train_steps is not None:
        epochs = int(np.ceil(max_train_steps / steps_per_epoch))
        hp.nepochs = epochs
        print("info; Number of epochs is set based on max_train_steps: {}".format(epochs))


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0, constant_values=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=constant_values)
    return x

# from: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py
# to avoid keras dependency


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# TODO: I know this is too ugly...
class _NPYDataSource(FileDataSource):
    def __init__(self, dump_root, col, typ="", speaker_id=None, max_steps=8000,
                 cin_pad=0, hop_size=256):
        self.dump_root = dump_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.max_steps = max_steps
        self.cin_pad = cin_pad
        self.hop_size = hop_size
        self.typ = typ

    def collect_files(self):
        meta = join(self.dump_root, "train.txt")
        if not exists(meta):
            paths = sorted(glob(join(self.dump_root, "*-{}.npy".format(self.typ))))
            return paths

        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.dump_root, f), paths_relative))

        # Exclude small files (assuming lenghts are in frame unit)
        # TODO: consider this for multi-speaker
        if self.max_steps is not None:
            idx = np.array(self.lengths) * self.hop_size > self.max_steps + 2 * self.cin_pad * self.hop_size
            if idx.sum() != len(self.lengths):
                print("{} short samples are omitted for training.".format(len(self.lengths) - idx.sum()))
            self.lengths = list(np.array(self.lengths)[idx])
            paths = list(np.array(paths)[idx])

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])
                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

        if self.multi_speaker:
            speaker_ids_np = list(np.array(self.speaker_ids)[indices])
            self.speaker_ids = list(map(int, speaker_ids_np))
            assert len(paths) == len(self.speaker_ids)

        return paths

    def collect_features(self, path):
        return np.load(path)


class RawAudioDataSource(_NPYDataSource):
    def __init__(self, dump_root, **kwargs):
        super(RawAudioDataSource, self).__init__(dump_root, 0, "wave", **kwargs)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, dump_root, **kwargs):
        super(MelSpecDataSource, self).__init__(dump_root, 1, "feats", **kwargs)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, batch_size=8, batch_group_size=None):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 8, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0

    def __iter__(self):
        indices = self.sorted_indices.numpy()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        bins = []
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            group = indices[s:e]
            random.shuffle(group)
            bins += [group]

        # Permutate batches
        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            last_bin = indices[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.sorted_indices)


class PyTorchDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.Mel is None:
            mel = None
        else:
            mel = self.Mel[idx]

        raw_audio = self.X[idx]
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        # (x,c,g)
        return raw_audio, mel, speaker_id

    def __len__(self):
        return len(self.X)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def clone_as_averaged_model(device, model, ema):
    assert ema is not None
    averaged_model = build_model().to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(target)
        losses = self.criterion(input, target)
        return ((losses * mask_).sum()) / mask_.sum()


class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self):
        super(DiscretizedMixturelogisticLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, 1)
        mask_ = mask.expand_as(target)

        losses = discretized_mix_logistic_loss(
            input, target, num_classes=hparams.quantize_channels,
            log_scale_min=hparams.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return ((losses * mask_).sum()) / mask_.sum()


class MixtureGaussianLoss(nn.Module):
    def __init__(self):
        super(MixtureGaussianLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, 1)
        mask_ = mask.expand_as(target)

        losses = mix_gaussian_loss(
            input, target, log_scale_min=hparams.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return ((losses * mask_).sum()) / mask_.sum()


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c, cin_pad):
    assert len(x) == (len(c) - 2 * cin_pad) * audio.get_hop_size()


def collate_fn(batch):
    """Create batch

    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T,)
            - x[1] (ndarray,int) : list of (T, D)
            - x[2] (ndarray,int) : list of (1,), speaker id
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2 and hparams.cin_channels > 0
    global_conditioning = len(batch[0]) >= 3 and hparams.gin_channels > 0

    if hparams.max_time_sec is not None:
        max_time_steps = int(hparams.max_time_sec * hparams.sample_rate)
    elif hparams.max_time_steps is not None:
        max_time_steps = hparams.max_time_steps
    else:
        max_time_steps = None

    # Time resolution adjustment
    cin_pad = hparams.cin_pad
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            if hparams.upsample_conditional_features:
                assert_ready_for_upsampling(x, c, cin_pad=0)
                if max_time_steps is not None:
                    max_steps = ensure_divisible(max_time_steps, audio.get_hop_size(), True)
                    if len(x) > max_steps:
                        max_time_frames = max_steps // audio.get_hop_size()
                        s = np.random.randint(cin_pad, len(c) - max_time_frames - cin_pad)
                        ts = s * audio.get_hop_size()
                        x = x[ts:ts + audio.get_hop_size() * max_time_frames]
                        c = c[s - cin_pad:s + max_time_frames + cin_pad, :]
                        assert_ready_for_upsampling(x, c, cin_pad=cin_pad)
            else:
                x, c = audio.adjust_time_resolution(x, c)
                if max_time_steps is not None and len(x) > max_time_steps:
                    s = np.random.randint(cin_pad, len(x) - max_time_steps - cin_pad)
                    x = x[s:s + max_time_steps]
                    c = c[s - cin_pad:s + max_time_steps + cin_pad, :]
                assert len(x) == len(c)
            new_batch.append((x, c, g))
        batch = new_batch
    else:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            x = audio.trim(x)
            if max_time_steps is not None and len(x) > max_time_steps:
                s = np.random.randint(0, len(x) - max_time_steps)
                if local_conditioning:
                    x, c = x[s:s + max_time_steps], c[s:s + max_time_steps, :]
                else:
                    x = x[s:s + max_time_steps]
            new_batch.append((x, c, g))
        batch = new_batch

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # (B, T, C)
    # pad for time-axis
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels - 1)
        x_batch = np.array([_pad_2d(to_categorical(
            x[0], num_classes=hparams.quantize_channels),
            max_input_len, 0, padding_value) for x in batch], dtype=np.float32)
    else:
        x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len)
                            for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels - 1)
        y_batch = np.array([_pad(x[0], max_input_len, constant_values=padding_value)
                            for x in batch], dtype=np.int)
    else:
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    # (B, T, D)
    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T)
        c_batch = torch.FloatTensor(c_batch).transpose(1, 2).contiguous()
    else:
        c_batch = None

    if global_conditioning:
        g_batch = torch.LongTensor([x[2] for x in batch])
    else:
        g_batch = None

    # Covnert to channel first i.e., (B, C, T)
    x_batch = torch.FloatTensor(x_batch).transpose(1, 2).contiguous()
    # Add extra axis
    if is_mulaw_quantize(hparams.input_type):
        y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()
    else:
        y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, y_batch, c_batch, g_batch, input_lengths


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def save_waveplot(path, y_hat, y_target):
    sr = hparams.sample_rate

    plt.figure(figsize=(16, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(y_target, sr=sr)
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(y_hat, sr=sr)
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def eval_model(global_step, writer, device, model, y, c, g, input_lengths, eval_dir, ema=None):
    if ema is not None:
        print("Using averaged model for evaluation")
        model = clone_as_averaged_model(device, model, ema)
        model.make_generation_fast_()

    model.eval()
    idx = np.random.randint(0, len(y))
    length = input_lengths[idx].data.cpu().item()

    # (T,)
    y_target = y[idx].view(-1).data.cpu().numpy()[:length]

    if c is not None:
        if hparams.upsample_conditional_features:
            c = c[idx, :, :length // audio.get_hop_size() + hparams.cin_pad * 2].unsqueeze(0)
        else:
            c = c[idx, :, :length].unsqueeze(0)
        assert c.dim() == 3
        print("Shape of local conditioning features: {}".format(c.size()))
    if g is not None:
        # TODO: test
        g = g[idx]
        print("Shape of global conditioning features: {}".format(g.size()))

    # Dummy silence
    if is_mulaw_quantize(hparams.input_type):
        initial_value = P.mulaw_quantize(0, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        initial_value = P.mulaw(0.0, hparams.quantize_channels)
    else:
        initial_value = 0.0

    # (C,)
    if is_mulaw_quantize(hparams.input_type):
        initial_input = to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
    initial_input = initial_input.to(device)

    # Run the model in fast eval mode
    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, softmax=True, quantize=True, tqdm=tqdm,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels - 1)
        y_target = P.inv_mulaw_quantize(y_target, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
        y_target = P.inv_mulaw(y_target, hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    # Save audio
    os.makedirs(eval_dir, exist_ok=True)
    path = join(eval_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(eval_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y_target, sr=hparams.sample_rate)

    # save figure
    path = join(eval_dir, "step{:09d}_waveplots.png".format(global_step))
    save_waveplot(path, y_hat, y_target)


def save_states(global_step, writer, y_hat, y, input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))
    idx = np.random.randint(0, len(y_hat))
    length = input_lengths[idx].data.cpu().item()

    # (B, C, T)
    if y_hat.dim() == 4:
        y_hat = y_hat.squeeze(-1)

    if is_mulaw_quantize(hparams.input_type):
        # (B, T)
        y_hat = F.softmax(y_hat, dim=1).max(1)[1]

        # (T,)
        y_hat = y_hat[idx].data.cpu().long().numpy()
        y = y[idx].view(-1).data.cpu().long().numpy()

        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels - 1)
        y = P.inv_mulaw_quantize(y, hparams.quantize_channels - 1)
    else:
        # (B, T)
        if hparams.output_distribution == "Logistic":
            y_hat = sample_from_discretized_mix_logistic(
                y_hat, log_scale_min=hparams.log_scale_min)
        elif hparams.output_distribution == "Normal":
            y_hat = sample_from_mix_gaussian(
                y_hat, log_scale_min=hparams.log_scale_min)
        else:
            assert False

        # (T,)
        y_hat = y_hat[idx].view(-1).data.cpu().numpy()
        y = y[idx].view(-1).data.cpu().numpy()

        if is_mulaw(hparams.input_type):
            y_hat = P.inv_mulaw(y_hat, hparams.quantize_channels)
            y = P.inv_mulaw(y, hparams.quantize_channels)

    # Mask by length
    y_hat[length:] = 0
    y[length:] = 0

    # Save audio
    audio_dir = join(checkpoint_dir, "intermediate", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path = join(audio_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y, sr=hparams.sample_rate)

# workaround for https://github.com/pytorch/pytorch/issues/15716
# the idea is to return outputs and replicas explicitly, so that making pytorch
# not to release the nodes (this is a pytorch bug though)


def data_parallel_workaround(model, input):
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    return y_hat, outputs, replicas


def __train_step(device, phase, epoch, global_step, global_test_step,
                 model, optimizer, writer, criterion,
                 x, y, c, g, input_lengths,
                 checkpoint_dir, eval_dir=None, do_eval=False, ema=None):
    sanity_check(model, c, g)

    # x : (B, C, T)
    # y : (B, T, 1)
    # c : (B, C, T)
    # g : (B,)
    train = (phase == "train_no_dev")
    clip_thresh = hparams.clip_thresh
    if train:
        model.train()
        step = global_step
    else:
        model.eval()
        step = global_test_step

    # Learning rate schedule
    current_lr = hparams.optimizer_params["lr"]
    if train and hparams.lr_schedule is not None:
        lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
        current_lr = lr_schedule_f(
            hparams.optimizer_params["lr"], step, **hparams.lr_schedule_kwargs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    optimizer.zero_grad()

    # Prepare data
    x, y = x.to(device), y.to(device)
    input_lengths = input_lengths.to(device)
    c = c.to(device) if c is not None else None
    g = g.to(device) if g is not None else None

    # (B, T, 1)
    mask = sequence_mask(input_lengths, max_len=x.size(-1)).unsqueeze(-1)
    mask = mask[:, 1:, :]

    # Apply model: Run the model in regular eval mode
    # NOTE: softmax is handled in F.cross_entrypy_loss
    # y_hat: (B x C x T)

    if use_cuda:
        # multi gpu support
        # you must make sure that batch size % num gpu == 0
        y_hat, _outputs, _replicas = data_parallel_workaround(model, (x, c, g, False))
    else:
        y_hat = model(x, c, g, False)

    if is_mulaw_quantize(hparams.input_type):
        # wee need 4d inputs for spatial cross entropy loss
        # (B, C, T, 1)
        y_hat = y_hat.unsqueeze(-1)
        loss = criterion(y_hat[:, :, :-1, :], y[:, 1:, :], mask=mask)
    else:
        loss = criterion(y_hat[:, :, :-1], y[:, 1:, :], mask=mask)

    if train and step > 0 and step % hparams.checkpoint_interval == 0:
        save_states(step, writer, y_hat, y, input_lengths, checkpoint_dir)
        save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch, ema)

    if do_eval:
        # NOTE: use train step (i.e., global_step) for filename
        eval_model(global_step, writer, device, model, y, c, g, input_lengths, eval_dir, ema)

    # Update
    if train:
        loss.backward()
        if clip_thresh > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
        optimizer.step()
        # update moving average
        if ema is not None:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

    # Logs
    writer.add_scalar("{} loss".format(phase), float(loss.item()), step)
    if train:
        if clip_thresh > 0:
            writer.add_scalar("gradient norm", grad_norm, step)
        writer.add_scalar("learning rate", current_lr, step)

    return loss.item()


def train_loop(device, model, data_loaders, optimizer, writer, checkpoint_dir=None):
    if is_mulaw_quantize(hparams.input_type):
        criterion = MaskedCrossEntropyLoss()
    else:
        if hparams.output_distribution == "Logistic":
            criterion = DiscretizedMixturelogisticLoss()
        elif hparams.output_distribution == "Normal":
            criterion = MixtureGaussianLoss()
        else:
            raise RuntimeError(
                "Not supported output distribution type: {}".format(
                    hparams.output_distribution))

    if hparams.exponential_moving_average:
        ema = ExponentialMovingAverage(hparams.ema_decay)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

    global global_step, global_epoch, global_test_step
    while global_epoch < hparams.nepochs:
        for phase, data_loader in data_loaders.items():
            train = (phase == "train_no_dev")
            running_loss = 0.
            test_evaluated = False
            for step, (x, y, c, g, input_lengths) in tqdm(enumerate(data_loader)):
                # Whether to save eval (i.e., online decoding) result
                do_eval = False
                eval_dir = join(checkpoint_dir, "intermediate", "{}_eval".format(phase))
                # Do eval per eval_interval for train
                if train and global_step > 0 \
                        and global_step % hparams.train_eval_interval == 0:
                    do_eval = True
                # Do eval for test
                # NOTE: Decoding WaveNet is quite time consuming, so
                # do only once in a single epoch for testset
                if not train and not test_evaluated \
                        and global_epoch % hparams.test_eval_epoch_interval == 0:
                    do_eval = True
                    test_evaluated = True
                if do_eval:
                    print("[{}] Eval at train step {}".format(phase, global_step))

                # Do step
                running_loss += __train_step(device,
                                             phase, global_epoch, global_step, global_test_step, model,
                                             optimizer, writer, criterion, x, y, c, g, input_lengths,
                                             checkpoint_dir, eval_dir, do_eval, ema)

                # update global state
                if train:
                    global_step += 1
                else:
                    global_test_step += 1

                if global_step >= hparams.max_train_steps:
                    print("Training reached max train steps ({}). will exit".format(hparams.max_train_steps))
                    return ema

            # log per epoch
            averaged_loss = running_loss / len(data_loader)
            writer.add_scalar("{} loss (per epoch)".format(phase),
                              averaged_loss, global_epoch)
            print("Step {} [{}] Loss: {}".format(
                global_step, phase, running_loss / len(data_loader)))

        global_epoch += 1
    return ema


def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch, ema=None):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    import shutil
    latest_pth = join(checkpoint_dir, "checkpoint_latest.pth")
    shutil.copyfile(checkpoint_path, latest_pth)

    if ema is not None:
        averaged_model = clone_as_averaged_model(device, model, ema)
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({
            "state_dict": averaged_model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved averaged checkpoint:", checkpoint_path)

        latest_pth = join(checkpoint_dir, "checkpoint_latest_ema.pth")
        shutil.copyfile(checkpoint_path, latest_pth)


def build_model():
    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    upsample_params = hparams.upsample_params
    upsample_params["cin_channels"] = hparams.cin_channels
    upsample_params["cin_pad"] = hparams.cin_pad
    model = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
    )
    return model


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = _load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))


def get_data_loaders(dump_root, speaker_id, test_shuffle=True):
    data_loaders = {}
    local_conditioning = hparams.cin_channels > 0

    if hparams.max_time_steps is not None:
        max_steps = ensure_divisible(hparams.max_time_steps, audio.get_hop_size(), True)
    else:
        max_steps = None

    for phase in ["train_no_dev", "dev"]:
        train = phase == "train_no_dev"
        X = FileSourceDataset(
            RawAudioDataSource(join(dump_root, phase), speaker_id=speaker_id,
                               max_steps=max_steps, cin_pad=hparams.cin_pad,
                               hop_size=audio.get_hop_size()))
        if local_conditioning:
            Mel = FileSourceDataset(
                MelSpecDataSource(join(dump_root, phase), speaker_id=speaker_id,
                                  max_steps=max_steps, cin_pad=hparams.cin_pad,
                                  hop_size=audio.get_hop_size()))
            assert len(X) == len(Mel)
            print("Local conditioning enabled. Shape of a sample: {}.".format(
                Mel[0].shape))
        else:
            Mel = None
        print("[{}]: length of the dataset is {}".format(phase, len(X)))

        if train:
            lengths = np.array(X.file_data_source.lengths)
            # Prepare sampler
            sampler = PartialyRandomizedSimilarTimeLengthSampler(
                lengths, batch_size=hparams.batch_size)
            shuffle = False
            # make sure that there's no sorting bugs for https://github.com/r9y9/wavenet_vocoder/issues/130
            sampler_idx = np.asarray(sorted(list(map(lambda s: int(s), sampler))))
            assert (sampler_idx == np.arange(len(sampler_idx), dtype=np.int)).all()
        else:
            sampler = None
            shuffle = test_shuffle

        dataset = PyTorchDataset(X, Mel)
        data_loader = data_utils.DataLoader(
            dataset, batch_size=hparams.batch_size, drop_last=True,
            num_workers=hparams.num_workers, sampler=sampler, shuffle=shuffle,
            collate_fn=collate_fn, pin_memory=hparams.pin_memory)

        speaker_ids = {}
        if X.file_data_source.multi_speaker:
            for idx, (x, c, g) in enumerate(dataset):
                if g is not None:
                    try:
                        speaker_ids[g] += 1
                    except KeyError:
                        speaker_ids[g] = 1
            if len(speaker_ids) > 0:
                print("Speaker stats:", speaker_ids)

        data_loaders[phase] = data_loader

    return data_loaders


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_restore_parts = args["--restore-parts"]
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None
    preset = args["--preset"]

    dump_root = args["--dump-root"]
    if dump_root is None:
        dump_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())

    fs = hparams.sample_rate

    os.makedirs(checkpoint_dir, exist_ok=True)

    output_json_path = join(checkpoint_dir, "hparams.json")
    with open(output_json_path, "w") as f:
        json.dump(hparams.values(), f, indent=2)

    # Dataloader setup
    data_loaders = get_data_loaders(dump_root, speaker_id, test_shuffle=True)

    maybe_set_epochs_based_on_max_steps(hparams, len(data_loaders["train_no_dev"]))

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = build_model().to(device)

    receptive_field = model.receptive_field
    print("Receptive field (samples / ms): {} / {}".format(
        receptive_field, receptive_field / fs * 1000))

    from torch import optim
    Optimizer = getattr(optim, hparams.optimizer)
    optimizer = Optimizer(model.parameters(), **hparams.optimizer_params)

    if checkpoint_restore_parts is not None:
        restore_parts(checkpoint_restore_parts, model)

    # Load checkpoints
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    ema = None
    try:
        ema = train_loop(device, model, data_loaders, optimizer, writer,
                         checkpoint_dir=checkpoint_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        save_checkpoint(
            device, model, optimizer, global_step, checkpoint_dir, global_epoch, ema)

    print("Finished")

    sys.exit(0)
