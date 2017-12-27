# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from .version import __version__

import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from deepvoice3_pytorch.modules import Conv1d, Linear


def Conv1d1x1(in_channels, out_channels):
    return Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 dropout=1 - 0.95, padding=None, dilation=1, causal=True,
                 residual=True, *args, **kwargs):
        super(Conv1dGLU, self).__init__()
        self.dropout = dropout
        self.residual = residual
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size,
                           dropout=dropout, padding=padding, dilation=dilation,
                           *args, **kwargs)
        self.conv1x1 = Conv1d1x1(out_channels, out_channels)

    def forward(self, x, c=None):
        return self._forward(x, c, False)

    def incremental_forward(self, x, c=None):
        return self._forward(x, c, True)

    def _forward(self, x, c, is_incremental):
        """Forward

        Args:
            x : B x C x T
            c : TODO
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        x = F.tanh(a) * F.sigmoid(b)
        x = self.conv1x1(x)

        y = (x + residual) * math.sqrt(0.5) if self.residual else x

        return y, x

    def clear_buffer(self):
        self.conv.clear_buffer()


class WaveNet(nn.Module):
    """WaveNet
    """

    def __init__(self, labels=256, channels=64, layers=12, stacks=2,
                 kernel_size=3, dropout=1 - 0.95):
        super(WaveNet, self).__init__()
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        C = channels
        self.first_conv = Conv1d1x1(labels, C)
        self.conv_layers = nn.ModuleList()
        for stack in range(stacks):
            for layer in range(layers_per_stack):
                conv = Conv1dGLU(C, C, kernel_size=kernel_size,
                                 dilation=2**layer, dropout=dropout)
                self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(),
            Conv1d1x1(C, C),
            nn.ReLU(),
            Conv1d1x1(C, labels),
        ])

    def forward(self, x):
        """Forward step

        Args:
            x : Variable of one-hot encoded audio signal, shape (B x labels x T)

        Returns:
            Variable: outupt, shape B x labels x T
        """
        x = self.first_conv(x)
        skips = None
        for f in self.conv_layers:
            x, h = f(x)
            skips = h if skips is None else (skips + h) * math.sqrt(0.5)

        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x
