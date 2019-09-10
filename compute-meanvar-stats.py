# coding: utf-8
"""Compute mean-variance normalization stats.

usage: compute_meanvar_stats.py [options] <list_file> <out_path>

options:
    -h, --help               Show help message.
    --verbose=<n>            Verbosity [default: 0].
"""
from docopt import docopt
import sys
from tqdm import tqdm
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == "__main__":
    args = docopt(__doc__)
    list_file = args["<list_file>"]
    out_path = args["<out_path>"]
    verbose = int(args["--verbose"])

    scaler = StandardScaler()
    with open(list_file) as f:
        lines = f.readlines()
    assert len(lines) > 0
    for path in tqdm(lines):
        c = np.load(path.strip())
        scaler.partial_fit(c)
    joblib.dump(scaler, out_path)

    if verbose > 0:
        print("mean:\n{}".format(scaler.mean_))
        print("var:\n{}".format(scaler.var_))

    sys.exit(0)
