# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from wavenet_vocoder import Conv1dGLU, WaveNet


def test_conv_block():
    conv = Conv1dGLU(30, 30, kernel_size=3, dropout=1 - 0.95)
    print(conv)
    x = Variable(torch.zeros(16, 30, 16000))
    y, h = conv(x)
    print(y.size(), h.size())


def test_wavenet():
    model = WaveNet()
    x = Variable(torch.zeros(16, 1, 1000))
    print(model)
    y = model(x)
    print(y.size())
