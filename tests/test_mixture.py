# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
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
    x = Variable(torch.rand(2, 16000, 30))
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
    y = Variable(torch.from_numpy(x)).float()
    y_hat = Variable(torch.rand(1, 30, T)).float()

    print(y.shape, y_hat.shape)

    loss = discretized_mix_logistic_loss(y_hat, y)
    print(loss)

    loss = discretized_mix_logistic_loss(y_hat, y, reduce=False)
    print(loss.size(), y.size())
    assert loss.size() == y.size()

    y = sample_from_discretized_mix_logistic(y_hat)
    print(y.shape)
