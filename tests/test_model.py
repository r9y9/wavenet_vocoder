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

import tensorflow as tf
# tf.set_verbosity

from keras.utils import np_utils
from wavenet_vocoder import Conv1dGLU, WaveNet

use_cuda = torch.cuda.is_available()
use_cuda = False


def test_conv_block():
    conv = Conv1dGLU(30, 30, kernel_size=3, dropout=1 - 0.95)
    print(conv)
    x = Variable(torch.zeros(16, 30, 16000))
    y, h = conv(x)
    print(y.size(), h.size())


def test_wavenet():
    model = WaveNet()
    x = Variable(torch.zeros(16, 256, 1000))
    y = model(x)
    print(y.size())


def test_incremental_forward_correctness():
    model = WaveNet(layers=20, stacks=2, channels=64)

    checkpoint_path = join(dirname(__file__), "..", "checkpoints/checkpoint_step000030000.pth")
    if exists(checkpoint_path):
        print("Loading from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    if use_cuda:
        model = model.cuda()

    sr = 4000
    x, _ = librosa.load(pysptk.util.example_audio_file(), sr=sr)
    x, _ = librosa.effects.trim(x, top_db=15)

    # To save computational cost
    x = x[:3000]

    x = P.mulaw_quantize(x)
    x_org = P.inv_mulaw_quantize(x)

    # (C, T)
    x = np_utils.to_categorical(x, num_classes=256).T
    # (1, C, T)
    x = x.reshape(1, 256, -1).astype(np.float32)
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

    # With zero start
    initial_input = x[:, :, 0].unsqueeze(-1).transpose(1, 2).contiguous()
    print(initial_input.size())
    print("Inital value:", initial_input.view(-1).max(0)[1])
    y_inference = model.incremental_forward(
        initial_input=initial_input, T=x.size(-1), tqdm=tqdm, softmax=True, quantize=True)

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

    save_audio = True
    if save_audio:
        librosa.output.write_wav("target.wav", x_org, sr=sr)
        librosa.output.write_wav("online.wav", y_online, sr=sr)
        librosa.output.write_wav("inference.wav", y_inference, sr=sr)
