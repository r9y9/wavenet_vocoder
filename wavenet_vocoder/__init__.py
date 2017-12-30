# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from .version import __version__

import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .modules import Conv1d1x1, Conv1dGLU


class WaveNet(nn.Module):
    """WaveNet
    """

    def __init__(self, labels=256, channels=64, layers=12, stacks=2,
                 kernel_size=3, dropout=1 - 0.95):
        super(WaveNet, self).__init__()
        self.labels = labels
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        C = channels
        self.first_conv = Conv1d1x1(labels, C)
        self.conv_layers = nn.ModuleList()
        for stack in range(stacks):
            for layer in range(layers_per_stack):
                dilation = 2**layer
                conv = Conv1dGLU(C, C, kernel_size=kernel_size,
                                 bias=True,  # magenda uses bias, but musyoku doesn't
                                 dilation=dilation, dropout=dropout)
                self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            Conv1d1x1(C, C),
            nn.ReLU(inplace=True),
            Conv1d1x1(C, labels),
        ])

    def forward(self, x, softmax=False):
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

        x = F.softmax(x, dim=1) if softmax else x

        return x

    def incremental_forward(self, initial_input=None, T=100, test_inputs=None,
                            tqdm=lambda x: x, softmax=True, quantize=True):
        self.clear_buffer()
        B = 1

        # Note: shape should be (B x T x C), not (B x C x T) opposed to batch forward
        # due to linealized convolution
        if test_inputs is not None:
            if test_inputs.size(1) == self.labels:
                test_inputs = test_inputs.transpose(1, 2).contiguous()
            B = test_inputs.size(0)
            T = max(T, test_inputs.size(1))

        outputs = []
        if initial_input is None:
            initial_input = Variable(torch.zeros(B, 1, self.labels))
            initial_input[:, :, 127] = 1
            # https://github.com/pytorch/pytorch/issues/584#issuecomment-275169567
            if next(self.parameters()).is_cuda:
                initial_input = initial_input.cuda()
        current_input = initial_input
        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.size(1):
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]

            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = None
            for f in self.conv_layers:
                x, h = f.incremental_forward(x)
                skips = h if skips is None else (skips + h) * math.sqrt(0.5)
            x = skips
            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x)
                except AttributeError:
                    x = f(x)

            x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
            if quantize:
                sample = np.random.choice(np.arange(self.labels), p=x.view(-1).data.cpu().numpy())
                x.zero_()
                x[:, sample] = 1.0
            outputs += [x]

        # T x B x C
        outputs = torch.stack(outputs)
        # B x C x T
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass
