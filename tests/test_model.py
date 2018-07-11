# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from nnmnkwii import preprocessing as P
from pysptk.util import example_audio_file
import librosa
import numpy as np
from tqdm import tqdm
from os.path import join, dirname, exists
from functools import partial
from nose.plugins.attrib import attr

from wavenet_vocoder.modules import ResidualConv1dGLU
from wavenet_vocoder import WaveNet

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

# For test
build_compact_model = partial(WaveNet, layers=4, stacks=2, residual_channels=32,
                              gate_channels=32, skip_out_channels=32,
                              scalar_input=False)

# https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py
# copied to avoid keras dependency in tests


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def test_conv_block():
    conv = ResidualConv1dGLU(30, 30, kernel_size=3, dropout=1 - 0.95)
    print(conv)
    x = torch.zeros(16, 30, 16000)
    y, h = conv(x)
    print(y.size(), h.size())


def test_wavenet_legacy():
    model = build_compact_model(legacy=True)
    print(model)
    x = torch.zeros(16, 256, 1000)
    y = model(x)
    print(y.size())


def test_wavenet():
    model = build_compact_model(legacy=False)
    print(model)
    x = torch.zeros(16, 256, 1000)
    y = model(x)
    print(y.size())


def _test_data(sr=4000, N=3000, returns_power=False, mulaw=True):
    x, _ = librosa.load(example_audio_file(), sr=sr)
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
    if mulaw:
        x = P.mulaw_quantize(x)
        x_org = P.inv_mulaw_quantize(x)
        # (C, T)
        x = to_categorical(x, num_classes=256).T
        # (1, C, T)
        x = x.reshape(1, 256, -1).astype(np.float32)
    else:
        x_org = x
        x = x.reshape(1, 1, -1)

    if returns_power:
        return x, x_org, p

    return x, x_org


@attr("mixture")
def test_mixture_wavenet():
    x, x_org, c = _test_data(returns_power=True, mulaw=False)
    # 10 mixtures
    model = build_compact_model(out_channels=3 * 10, cin_channels=1,
                                scalar_input=True)
    T = x.shape[-1]
    print(model.first_conv)

    # scalar input, not one-hot
    assert x.shape[1] == 1

    x = torch.from_numpy(x).contiguous().to(device)

    c = torch.from_numpy(c).contiguous().to(device)
    print(c.size())

    model.eval()

    # Incremental forward with forced teaching
    y_online = model.incremental_forward(
        test_inputs=x, c=c, T=None, tqdm=tqdm)

    assert y_online.size() == x.size()

    y_online2 = model.incremental_forward(
        test_inputs=None, c=c, T=T, tqdm=tqdm)

    assert y_online2.size() == x.size()
    print(x.size())


@attr("local_conditioning")
def test_local_conditioning_correctness():
    # condition by power
    x, x_org, c = _test_data(returns_power=True)
    model = build_compact_model(cin_channels=1)
    assert model.local_conditioning_enabled()
    assert not model.has_speaker_embedding()

    x = torch.from_numpy(x).contiguous().to(device)

    c = torch.from_numpy(c).contiguous().to(device)
    print(x.size(), c.size())

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
    except Exception:
        from warnings import warn
        warn("oops! must be a bug!")


@attr("local_conditioning")
def test_local_conditioning_upsample_correctness():
    # condition by power
    x, x_org, c = _test_data(returns_power=True)

    # downsample by 4
    assert c.shape[-1] % 4 == 0
    c = c[:, :, 0::4]

    model = build_compact_model(
        cin_channels=1, upsample_conditional_features=True,
        upsample_scales=[2, 2])
    assert model.local_conditioning_enabled()
    assert not model.has_speaker_embedding()

    x = torch.from_numpy(x).contiguous().to(device)

    c = torch.from_numpy(c).contiguous().to(device)
    print(x.size(), c.size())

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
    except Exception:
        from warnings import warn
        warn("oops! must be a bug!")


@attr("global_conditioning")
def test_global_conditioning_with_embedding_correctness():
    # condition by mean power
    x, x_org, c = _test_data(returns_power=True)
    g = c.mean(axis=-1, keepdims=True).astype(np.int)
    model = build_compact_model(gin_channels=16, n_speakers=256,
                                use_speaker_embedding=True)
    assert not model.local_conditioning_enabled()
    assert model.has_speaker_embedding()

    x = torch.from_numpy(x).contiguous().to(device)

    g = torch.from_numpy(g).long().contiguous().to(device)
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
    except Exception:
        from warnings import warn
        warn("oops! must be a bug!")


@attr("global_conditioning")
def test_global_conditioning_correctness():
    # condition by mean power
    x, x_org, c = _test_data(returns_power=True)
    # must be floating-point type
    g = c.mean(axis=-1, keepdims=True).astype(np.float32)
    model = build_compact_model(gin_channels=1, use_speaker_embedding=False)
    assert not model.local_conditioning_enabled()
    # `use_speaker_embedding` False should diable embedding layer
    assert not model.has_speaker_embedding()

    x = torch.from_numpy(x).contiguous().to(device)

    g = torch.from_numpy(g).contiguous().to(device)
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
    except Exception:
        from warnings import warn
        warn("oops! must be a bug!")


@attr("local_and_global_conditioning")
def test_global_and_local_conditioning_correctness():
    x, x_org, c = _test_data(returns_power=True)
    g = c.mean(axis=-1, keepdims=True).astype(np.int)
    model = build_compact_model(cin_channels=1, gin_channels=16, n_speakers=256)
    assert model.local_conditioning_enabled()
    assert model.has_speaker_embedding()

    x = torch.from_numpy(x).contiguous().to(device)

    # per-sample power
    c = torch.from_numpy(c).contiguous().to(device)

    # mean power
    g = torch.from_numpy(g).long().contiguous().to(device)

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
    except Exception:
        from warnings import warn
        warn("oops! must be a bug!")


@attr("local_only")
def test_incremental_forward_correctness():
    import librosa.display
    from matplotlib import pyplot as plt

    model = build_compact_model().to(device)

    checkpoint_path = join(dirname(__file__), "..", "foobar/checkpoint_step000058000.pth")
    if exists(checkpoint_path):
        print("Loading from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    sr = 4000
    x, x_org = _test_data(sr=sr, N=3000)
    x = torch.from_numpy(x).contiguous().to(device)

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
    except Exception:
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
