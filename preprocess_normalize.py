# coding: utf-8
"""Perform meanvar normalization to preprocessed features.

usage: preprocess_normalize.py [options] <in_dir> <out_dir> <scaler>

options:
    --inverse                Inverse transform.
    --num_workers=<n>        Num workers.
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from os.path import join, exists, basename, splitext
from multiprocessing import cpu_count
from tqdm import tqdm
from nnmnkwii import preprocessing as P
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from shutil import copyfile

import joblib
from glob import glob
from itertools import zip_longest


def get_paths_by_glob(in_dir, filt):
    return glob(join(in_dir, filt))


def _process_utterance(out_dir, audio_path, feat_path, scaler, inverse):
    # [Optional] copy audio with the same name if exists
    if audio_path is not None and exists(audio_path):
        name = splitext(basename(audio_path))[0]
        np.save(join(out_dir, name), np.load(audio_path), allow_pickle=False)

    # [Required] apply normalization for features
    assert exists(feat_path)
    x = np.load(feat_path)
    if inverse:
        y = scaler.inverse_transform(x)
    else:
        y = scaler.transform(x)
    assert x.dtype == y.dtype
    name = splitext(basename(feat_path))[0]
    np.save(join(out_dir, name), y, allow_pickle=False)


def apply_normalization_dir2dir(in_dir, out_dir, scaler, inverse, num_workers):
    # NOTE: at this point, audio_paths can be empty
    audio_paths = get_paths_by_glob(in_dir, "*-wave.npy")
    feature_paths = get_paths_by_glob(in_dir, "*-feats.npy")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for audio_path, feature_path in zip_longest(audio_paths, feature_paths):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, audio_path, feature_path, scaler, inverse)))
    for future in tqdm(futures):
        future.result()


if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    scaler_path = args["<scaler>"]
    scaler = joblib.load(scaler_path)
    inverse = args["--inverse"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() // 2 if num_workers is None else int(num_workers)

    os.makedirs(out_dir, exist_ok=True)
    apply_normalization_dir2dir(in_dir, out_dir, scaler, inverse, num_workers)

    # Copy meta information if exists
    traintxt = join(in_dir, "train.txt")
    if exists(traintxt):
        copyfile(join(in_dir, "train.txt"), join(out_dir, "train.txt"))
