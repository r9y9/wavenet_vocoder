# coding: utf-8
"""
Dump hyper parameters to json file.

usage: tojson.py [options] <output_json_path>

options:
    --hparams=<parmas>       Hyper parameters [default: ].
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import json

from hparams import hparams

if __name__ == "__main__":
    args = docopt(__doc__)
    output_json_path = args["<output_json_path>"]

    hparams.parse(args["--hparams"])
    j = hparams.values()
    with open(output_json_path, "w") as f:
        json.dump(j, f, indent=2)
    sys.exit(0)
