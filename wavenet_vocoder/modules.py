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
    return Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=True)


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit
    """

    def __init__(self, in_channels, out_channels, kernel_size,
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
        self.conv1x1_out = Conv1d1x1(out_channels, out_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(out_channels, out_channels, bias=bias)

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

        # For skip connection
        if is_incremental:
            s = self.conv1x1_skip.incremental_forward(x)
        else:
            s = self.conv1x1_skip(x)

        # For residual connection
        if is_incremental:
            x = self.conv1x1_out.incremental_forward(x)
        else:
            x = self.conv1x1_out(x)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for conv in [self.conv, self.conv1x1_out, self.conv1x1_skip]:
            self.conv.clear_buffer()
