# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import librosa
import pysptk

from wavenet_vocoder.mixture import discretized_mix_logistic_loss
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=-1, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def test_log_softmax():
    x = torch.rand(2, 16000, 30)
    y = log_prob_from_logits(x)
    y_hat = F.log_softmax(x, -1)

    y = y.data.cpu().numpy()
    y_hat = y_hat.data.cpu().numpy()
    assert np.allclose(y, y_hat)


def test_mixture():
    np.random.seed(1234)

    x, sr = librosa.load(pysptk.util.example_audio_file(), sr=None)
    assert sr == 16000

    T = len(x)
    x = x.reshape(1, T, 1)
    y = torch.from_numpy(x).float()
    y_hat = torch.rand(1, 30, T).float()

    print(y.shape, y_hat.shape)

    loss = discretized_mix_logistic_loss(y_hat, y)
    print(loss)

    loss = discretized_mix_logistic_loss(y_hat, y, reduce=False)
    print(loss.size(), y.size())
    assert loss.size() == y.size()

    y = sample_from_discretized_mix_logistic(y_hat)
    print(y.shape)


def test_misc():
    # https://en.wikipedia.org/wiki/Logistic_distribution
    # what i have learned
    # m = (x - mu) / s
    m = torch.rand(10, 10)
    log_pdf_mid1 = -2 * torch.log(torch.exp(m / 2) + torch.exp(-m / 2))
    log_pdf_mid2 = m - 2 * F.softplus(m)
    assert np.allclose(log_pdf_mid1.data.numpy(), log_pdf_mid2.data.numpy())

    # Edge case for 0
    plus_in = torch.rand(10, 10)
    log_cdf_plus1 = torch.sigmoid(m).log()
    log_cdf_plus2 = m - F.softplus(m)
    assert np.allclose(log_cdf_plus1.data.numpy(), log_cdf_plus2.data.numpy())

    # Edge case for 255
    min_in = torch.rand(10, 10)
    log_one_minus_cdf_min1 = (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min2 = -F.softplus(min_in)
    assert np.allclose(log_one_minus_cdf_min1.data.numpy(), log_one_minus_cdf_min2.data.numpy())
