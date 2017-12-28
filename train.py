"""Trainining script for seq2seq text-to-speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --load-embedding=<path>      Load embedding from checkpoint.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt

import sys
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime

from wavenet_vocoder import builder
import lrschedule

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np
from numba import jit

from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii.datasets import cmu_arctic

from os.path import join, expanduser
import random
from scipy.io import wavfile
import librosa.display
from matplotlib import pyplot as plt
import sys
import os


from keras.utils import np_utils
from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn
from hparams import hparams, hparams_debug_string

fs = hparams.sample_rate

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

_frontend = None  # to be set later


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


class CMUArcticWavDataSource(cmu_arctic.WavFileDataSource):
    def __init__(self, data_root, speakers=["clb"]):
        super(CMUArcticWavDataSource, self).__init__(data_root, speakers)

    def collect_features(self, path):
        x, _ = librosa.load(path, sr=hparams.sample_rate)
        x, _ = librosa.effects.trim(x, top_db=30)
        # (T,)
        x = P.mulaw_quantize(x)
        return x


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randmoized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batchs
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


# TODO
class PyTorchDataset(object):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        return self.X[idx],

    def __len__(self):
        return len(self.X)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

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


def collate_fn(batch):
    """Create batch

    Args:
        batch(tuple): Tuple of inputs
            - x[0] (ndarray,int) : list of (T,)
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """
    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # (B, T, C)
    # pad for time-axis
    x_batch = np.array([_pad_2d(np_utils.to_categorical(x[0], num_classes=256),
                                max_input_len) for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    assert len(y_batch.shape) == 2

    # Covnert to channel first i.e., (B, C, T)
    x_batch = torch.FloatTensor(x_batch).transpose(1, 2).contiguous()
    # Add extra axis
    y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, y_batch, input_lengths


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def eval_model(global_step, writer, model, y, input_lengths, checkpoint_dir):
    print("Eval model at step {}".format(global_step))
    model.eval()
    idx = np.random.randint(0, len(y))
    length = input_lengths[idx].data.cpu().numpy()[0]

    # (T,)
    y_target = y[idx].view(-1).data.cpu().long().numpy()[:length]

    # (C,)
    initial_input = np_utils.to_categorical(y_target[0], num_classes=256).astype(np.float32)
    initial_input = Variable(torch.from_numpy(initial_input)).view(1, 1, 256)
    initial_input = initial_input.cuda() if use_cuda else initial_input
    y_hat = model.incremental_forward(initial_input, T=length, tqdm=tqdm)
    y_hat = F.softmax(y_hat, dim=1).max(1)[1].view(-1).long().cpu().data.numpy()
    y_hat = P.inv_mulaw_quantize(y_hat)

    y_target = P.inv_mulaw_quantize(y_target)

    # Save audio
    audio_dir = join(checkpoint_dir, "eval")
    os.makedirs(audio_dir, exist_ok=True)
    path = join(audio_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y_target, sr=hparams.sample_rate)


def save_states(global_step, writer, y_hat, y, input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))
    idx = np.random.randint(0, len(y_hat))
    length = input_lengths[idx].data.cpu().numpy()[0]

    # (B, C, T)
    y_hat = y_hat.squeeze(-1)
    # (B, T)
    y_hat = F.softmax(y_hat, dim=1).max(1)[1]

    # (T,)
    y_hat = y_hat[idx].data.cpu().long().numpy()
    y = y[idx].view(-1).data.cpu().long().numpy()

    y_hat = P.inv_mulaw_quantize(y_hat)
    y = P.inv_mulaw_quantize(y)

    # Mask by length
    y_hat[length:] = 0
    y[length:] = 0

    # Save audio
    audio_dir = join(checkpoint_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path = join(audio_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y, sr=hparams.sample_rate)


def train(model, data_loader, optimizer, writer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    if use_cuda:
        model = model.cuda()
    current_lr = init_lr

    criterion = MaskedCrossEntropyLoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, y, input_lengths) in tqdm(enumerate(data_loader)):
            # x : (B, C, T)
            # y : (B, T, 1)
            model.train()
            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(
                    init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            # Prepare data
            x, y = Variable(x), Variable(y)
            input_lengths = Variable(input_lengths)
            if use_cuda:
                x, y = x.cuda(), y.cuda()
                input_lengths = input_lengths.cuda()

            # (B, T, 1)
            mask = sequence_mask(input_lengths, max_len=x.size(-1)).unsqueeze(-1)
            mask = mask[:, 1:, :]

            # Apply model
            y_hat = model(x)

            # wee need 4d inputs for spatial cross entropy loss
            # (B, C, T, 1)
            y_hat = y_hat.unsqueeze(-1)

            loss = criterion(y_hat[:, :, :-1, :], y[:, 1:, :], mask=mask)

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states(
                    global_step, writer, y_hat, y, input_lengths, checkpoint_dir)
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step > 0 and global_step % hparams.eval_interval == 0:
                eval_model(global_step, writer, model, y, input_lengths, checkpoint_dir)

            # Update
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            writer.add_scalar("loss", float(loss.data[0]), global_step)
            if clip_thresh > 0:
                writer.add_scalar("gradient norm", grad_norm, global_step)
            writer.add_scalar("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.data[0]

        averaged_loss = running_loss / (len(data_loader))
        writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss / (len(data_loader))))

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def build_model():
    model = getattr(builder, hparams.builder)(
        layers=hparams.layers,
        stacks=hparams.stacks,
        channels=hparams.channels,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size)
    return model


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = torch.load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}
    model_dict.update(valid_state_dict)
    model.load_state_dict(model_dict)


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_restore_parts = args["--restore-parts"]

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    print(hparams_debug_string())
    assert hparams.name == "wavenet_vocoder"

    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json
        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(CMUArcticWavDataSource(data_root))
    print(len(X))

    # TODO: better way
    lengths = []
    for i in tqdm(range(len(X))):
        lengths.append(X[i].shape[0])
    lengths = np.array(lengths)

    # Prepare sampler
    sampler = PartialyRandomizedSimilarTimeLengthSampler(
        lengths, batch_size=hparams.batch_size)

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, sampler=sampler,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = build_model()
    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay)

    if checkpoint_restore_parts is not None:
        restore_parts(checkpoint_restore_parts, model)

    # Load checkpoints
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("Los event path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    try:
        train(model, data_loader, optimizer, writer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              clip_thresh=hparams.clip_thresh,
              )
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)
