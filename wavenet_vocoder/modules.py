# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from deepvoice3_pytorch.modules import Conv1d


def Conv1d1x1(in_channels, out_channels, bias=True):
    """1-by-1 convolution layer
    """
    return Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=True)


def _conv1x1_forward(conv, x, is_incremental):
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 cin_channels=None, gin_channels=None,
                 dropout=1 - 0.95, padding=None, dilation=1, causal=True,
                 bias=True, *args, **kwargs):
        super(Conv1dGLU, self).__init__()
        self.dropout = dropout
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size,
                           dropout=dropout, padding=padding, dilation=dilation,
                           bias=bias, *args, **kwargs)
        # local conditioning
        if cin_channels is not None:
            self.conv1x1c = Conv1d1x1(cin_channels, 2 * out_channels, bias=bias)
        else:
            self.conv1x1c = None

        # global conditioning
        if gin_channels is not None:
            self.conv1x1g = Conv1d1x1(gin_channels, 2 * out_channels, bias=bias)
        else:
            self.conv1x1g = None

        self.conv1x1_out = Conv1d1x1(out_channels, out_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(out_channels, out_channels, bias=bias)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """Forward

        Args:
            x (Variable): B x C x T
            c (Variable): B x C x T, Local conditioning features
            g (Variable): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not

        Returns:
            Variable: output
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

        # local conditioning
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb

        x = F.tanh(a) * F.sigmoid(b)

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for conv in [self.conv, self.conv1x1_out, self.conv1x1_skip]:
            self.conv.clear_buffer()
