# coding: utf-8
import torch
from torch import nn


def wavenet(layers=20,
            stacks=2,
            channels=256,
            skip_out_channels=512,
            cin_channels=None,
            gin_channels=None,
            weight_normalization=True,
            dropout=1 - 0.95,
            kernel_size=3,
            n_speakers=None,
            ):
    from wavenet_vocoder import WaveNet

    model = WaveNet(layers=layers, stacks=stacks,
                    channels=channels, dropout=dropout,
                    skip_out_channels=skip_out_channels,
                    kernel_size=kernel_size,
                    weight_normalization=weight_normalization,
                    cin_channels=cin_channels, gin_channels=gin_channels,
                    n_speakers=n_speakers)

    return model
