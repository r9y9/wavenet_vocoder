# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import sys
from os.path import dirname, join
sys.path.insert(0, join(dirname(__file__), ".."))

import numpy as np
from nose.plugins.attrib import attr


@attr("local_only")
def test_amp_to_db():
    # this will invoke tensorflow initialization and you will see too many logs
    # which is annoying. Limit this only for local
    import audio

    x = np.random.rand(10)
    x_hat = audio.db_to_amp(audio._amp_to_db(x))
    assert (x == x_hat).all()
