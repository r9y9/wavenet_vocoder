# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from wavenet_vocoder import receptive_field_size


def test_receptive_field_size():
    # Table 4 in https://arxiv.org/abs/1711.10433
    assert receptive_field_size(total_layers=30, num_cycles=3, kernel_size=3) == 6139
    assert receptive_field_size(total_layers=24, num_cycles=4, kernel_size=3) == 505
    assert receptive_field_size(total_layers=12, num_cycles=2, kernel_size=3) == 253
    assert receptive_field_size(total_layers=30, num_cycles=1,
                                kernel_size=3, dilation=lambda x: 1) == 61
