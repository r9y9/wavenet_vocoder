# coding: utf-8
import torch
from torch import nn


def wavenet(channels=64,
            dropout=1 - 0.95,
            kernel_size=3,
            ):
    from wavenet_vocoder import WaveNet
    return WaveNet(channels=channels, dropout=dropout, kernel_size=kernel_size)
