# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import pysptk
from nnmnkwii import preprocessing as P
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

# https://github.com/tensorflow/tensorflow/issues/8340
import logging
logging.getLogger('tensorflow').disabled = True

from os.path import join, dirname, exists

from nose.plugins.attrib import attr

import tensorflow as tf
# tf.set_verbosity

from keras.utils import np_utils
from wavenet_vocoder import Conv1dGLU, WaveNet

use_cuda = torch.cuda.is_available()
use_cuda = False


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def test_conv_block():
    conv = Conv1dGLU(30, 30, kernel_size=3, dropout=1 - 0.95)
    print(conv)
    x = Variable(torch.zeros(16, 30, 16000))
    y, h = conv(x)
    print(y.size(), h.size())


def test_wavenet():
    model = WaveNet(layers=6, stacks=2, channels=32, kernel_size=2)
    x = Variable(torch.zeros(16, 256, 1000))
    y = model(x)
    print(y.size())


def _quantized_test_data(sr=4000, N=3000, returns_power=False):
    x, _ = librosa.load(pysptk.util.example_audio_file(), sr=sr)
    x, _ = librosa.effects.trim(x, top_db=15)

    # To save computational cost
    x = x[:N]

    # For power conditioning wavenet
    if returns_power:
        # (1 x N')
        p = librosa.feature.rmse(x, frame_length=256, hop_length=128)
        upsample_factor = x.size // p.size
        # (1 x N)
        p = np.repeat(p, upsample_factor, axis=-1)
        if p.size < x.size:
            # pad against time axis
            p = np.pad(p, [(0, 0), (0, x.size - p.size)], mode="constant", constant_values=0)

        # shape adajst
        p = p.reshape(1, 1, -1)

    # (T,)
    x = P.mulaw_quantize(x)
    x_org = P.inv_mulaw_quantize(x)

    # (C, T)
    x = np_utils.to_categorical(x, num_classes=256).T
    # (1, C, T)
    x = x.reshape(1, 256, -1).astype(np.float32)

    if returns_power:
        return x, x_org, p

    return x, x_org


@attr("local_conditioning")
def test_local_conditioning_correctness():
    # condition by power
    x, x_org, c = _quantized_test_data(returns_power=True)
    model = WaveNet(layers=6, stacks=2, channels=64, cin_channels=1)

    x = Variable(torch.from_numpy(x).contiguous())
    x = x.cuda() if use_cuda else x

    c = Variable(torch.from_numpy(c).contiguous())
    c = c.cuda() if use_cuda else c
    print(c.size())

    model.eval()

    y_offline = model(x, c=c, softmax=True)

    # Incremental forward with forced teaching
    y_online = model.incremental_forward(
        test_inputs=x, c=c, T=None, tqdm=tqdm, softmax=True, quantize=False)

    # (1 x C x T)
    c = (y_offline - y_online).abs()
    print(c.mean(), c.max())

    try:
        assert np.allclose(y_offline.cpu().data.numpy(),
                           y_online.cpu().data.numpy(), atol=1e-4)
    except:
        from warnings import warn
        warn("oops! must be a bug!")


@attr("global_conditioning")
def test_global_conditioning_correctness():
    # condition by mean power
    x, x_org, c = _quantized_test_data(returns_power=True)
    g = c.mean(axis=-1, keepdims=True).astype(np.int)
    model = WaveNet(layers=6, stacks=2, channels=64, gin_channels=16,
                    n_speakers=256)

    x = Variable(torch.from_numpy(x).contiguous())
    x = x.cuda() if use_cuda else x

    g = Variable(torch.from_numpy(g).contiguous())
    g = g.cuda() if use_cuda else g
    print(g.size())

    model.eval()

    y_offline = model(x, g=g, softmax=True)

    # Incremental forward with forced teaching
    y_online = model.incremental_forward(
        test_inputs=x, g=g, T=None, tqdm=tqdm, softmax=True, quantize=False)

    # (1 x C x T)
    c = (y_offline - y_online).abs()
    print(c.mean(), c.max())

    try:
        assert np.allclose(y_offline.cpu().data.numpy(),
                           y_online.cpu().data.numpy(), atol=1e-4)
    except:
        from warnings import warn
        warn("oops! must be a bug!")


@attr("local_and_global_conditioning")
def test_global_and_local_conditioning_correctness():
    x, x_org, c = _quantized_test_data(returns_power=True)
    g = c.mean(axis=-1, keepdims=True).astype(np.int)
    model = WaveNet(layers=6, stacks=2, channels=64,
                    cin_channels=1, gin_channels=16, n_speakers=256)

    x = Variable(torch.from_numpy(x).contiguous())
    x = x.cuda() if use_cuda else x

    # per-sample power
    c = Variable(torch.from_numpy(c).contiguous())
    c = c.cuda() if use_cuda else c

    # mean power
    g = Variable(torch.from_numpy(g).contiguous())
    g = g.cuda() if use_cuda else g

    print(c.size(), g.size())

    model.eval()

    y_offline = model(x, c=c, g=g, softmax=True)

    # Incremental forward with forced teaching
    y_online = model.incremental_forward(
        test_inputs=x, c=c, g=g, T=None, tqdm=tqdm, softmax=True, quantize=False)
    # (1 x C x T)

    c = (y_offline - y_online).abs()
    print(c.mean(), c.max())

    try:
        assert np.allclose(y_offline.cpu().data.numpy(),
                           y_online.cpu().data.numpy(), atol=1e-4)
    except:
        from warnings import warn
        warn("oops! must be a bug!")


def test_incremental_forward_correctness():
    model = WaveNet(layers=20, stacks=2, channels=128)

    checkpoint_path = join(dirname(__file__), "..", "foobar/checkpoint_step000058000.pth")
    if exists(checkpoint_path):
        print("Loading from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    if use_cuda:
        model = model.cuda()

    sr = 4000
    x, x_org = _quantized_test_data(sr=sr, N=3000)
    x = Variable(torch.from_numpy(x).contiguous())
    x = x.cuda() if use_cuda else x

    model.eval()

    # Batch forward
    y_offline = model(x, softmax=True)

    # Test from zero start
    y_online = model.incremental_forward(initial_input=None, T=100, tqdm=tqdm, softmax=True)

    # Incremental forward with forced teaching
    y_online = model.incremental_forward(test_inputs=x, tqdm=tqdm, softmax=True, quantize=False)

    # (1 x C x T)
    c = (y_offline - y_online).abs()
    print(c.mean(), c.max())

    try:
        assert np.allclose(y_offline.cpu().data.numpy(),
                           y_online.cpu().data.numpy(), atol=1e-4)
    except:
        from warnings import warn
        warn("oops! must be a bug!")

    # (1, T, C)
    xt = x.transpose(1, 2).contiguous()

    initial_input = xt[:, 0, :].unsqueeze(1).contiguous()
    print(initial_input.size())
    print("Inital value:", initial_input.view(-1).max(0)[1])

    # With zero start
    zerostart = True
    if zerostart:
        y_inference = model.incremental_forward(
            initial_input=initial_input, T=xt.size(1), tqdm=tqdm, softmax=True, quantize=True)
    else:
        # Feed a few samples as test_inputs and then generate auto-regressively
        N = 1000
        y_inference = model.incremental_forward(
            initial_input=None, test_inputs=xt[:, :N, :],
            T=xt.size(1), tqdm=tqdm, softmax=True, quantize=True)

    # Waveforms
    # (T,)
    y_offline = y_offline.max(1)[1].view(-1)
    y_online = y_online.max(1)[1].view(-1)
    y_inference = y_inference.max(1)[1].view(-1)

    y_offline = P.inv_mulaw_quantize(y_offline.cpu().data.long().numpy())
    y_online = P.inv_mulaw_quantize(y_online.cpu().data.long().numpy())
    y_inference = P.inv_mulaw_quantize(y_inference.cpu().data.long().numpy())

    plt.figure(figsize=(16, 10))
    plt.subplot(4, 1, 1)
    librosa.display.waveplot(x_org, sr=sr)
    plt.subplot(4, 1, 2)
    librosa.display.waveplot(y_offline, sr=sr)
    plt.subplot(4, 1, 3)
    librosa.display.waveplot(y_online, sr=sr)
    plt.subplot(4, 1, 4)
    librosa.display.waveplot(y_inference, sr=sr)
    plt.show()

    save_audio = False
    if save_audio:
        librosa.output.write_wav("target.wav", x_org, sr=sr)
        librosa.output.write_wav("online.wav", y_online, sr=sr)
        librosa.output.write_wav("inference.wav", y_inference, sr=sr)
