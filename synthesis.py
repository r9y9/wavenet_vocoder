# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.
    --conditional=<p>                 Conditional features path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
from torch.autograd import Variable
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa

import audio
from hparams import hparams


use_cuda = torch.cuda.is_available()


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, length=None, c=None, g=None, initial_value=None, fast=False):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    c = _to_numpy(c)
    g = _to_numpy(g)

    if use_cuda:
        model = model.cuda()
    model.eval()
    if fast:
        model.make_generation_fast_()

    if c is None:
        assert length is not None
    else:
        # (N, D)
        assert c.ndim == 2
        # (T, D)
        if not hparams.upsample_conditional_features:
            upsample_factor = audio.get_hop_size()
            c = np.repeat(c, upsample_factor, axis=0)
        length = c.shape[0]
        # B x C x T
        c = c.T.reshape(1, -1, length)
        c = Variable(torch.FloatTensor(c))

    if initial_value is None:
        initial_value = P.mulaw_quantize(0)  # dummy silence
    print("Initial value of the WaveNet decoder: {}".format(initial_value))
    assert initial_value >= 0 and initial_value < 256

    initial_input = np_utils.to_categorical(
        initial_value, num_classes=256).astype(np.float32)
    initial_input = Variable(torch.from_numpy(initial_input)).view(1, 1, 256)
    g = None if g is None else Variable(torch.LongTensor([g]))
    if use_cuda:
        initial_input = initial_input.cuda()
        g = None if g is None else g.cuda()
        c = None if c is None else c.cuda()

    y_hat = model.incremental_forward(
        initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True)
    y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
    y_hat = P.inv_mulaw_quantize(y_hat)

    return y_hat


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]

    length = int(args["--length"])
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else int(initial_value)
    conditional_path = args["--conditional"]
    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]
    speaker_id = args["--speaker_id"]
    speaker_id = None if speaker_id is None else int(speaker_id)

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    # Load conditional features
    if conditional_path is not None:
        c = np.load(conditional_path)
    else:
        c = None

    from train import build_model

    # Model
    model = build_model()

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    os.makedirs(dst_dir, exist_ok=True)
    dst_wav_path = join(dst_dir, "{}{}.wav".format(checkpoint_name, file_name_suffix))

    # DO generate
    waveform = wavegen(model, length, c=c, g=speaker_id, initial_value=initial_value, fast=True)

    # save
    librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
