# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import sys
from os.path import dirname, join
sys.path.insert(0, join(dirname(__file__), ".."))

import numpy as np
from nose.plugins.attrib import attr

import logging
logging.getLogger('tensorflow').disabled = True


@attr("local_only")
def test_amp_to_db():
    import audio
    x = np.random.rand(10)
    x_hat = audio._db_to_amp(audio._amp_to_db(x))
    assert np.allclose(x, x_hat)
