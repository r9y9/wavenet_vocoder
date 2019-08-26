# coding: utf-8
"""
Make subset of dataset

usage: mksubset.py [options] <in_dir> <out_dir>

options:
    -h, --help               Show help message.
    --limit=<N>              Limit dataset size by N-hours [default: 10000].
    --train-dev-test-split   Train/test split.
    --dev-size=<N>           Development size or rate [default: 0.1].
    --test-size=<N>          Test size or rate [default: 0.1].
    --target-sr=<N>          Resampling.
    --random-state=<N>       Random seed [default: 1234].
"""
from docopt import docopt
import librosa
from glob import glob
from os.path import join, basename, exists, splitext
from tqdm import tqdm
import sys
import os
from shutil import copy2
from scipy.io import wavfile
import numpy as np


def read_wav_or_raw(src_file, is_raw):
    if is_raw:
        sr = 24000  # hard coded for now
        x = np.fromfile(src_file, dtype=np.int16)
    else:
        sr, x = wavfile.read(src_file)
    return sr, x


def write_wav_or_raw(dst_path, sr, x, is_raw):
    if is_raw:
        x.tofile(dst_path)
    else:
        wavfile.write(dst_path, sr, x)

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    limit = float(args["--limit"])
    train_dev_test_split = args["--train-dev-test-split"]
    dev_size = float(args["--dev-size"])
    test_size = float(args["--test-size"])
    target_sr = args["--target-sr"]
    target_sr = int(target_sr) if target_sr is not None else None
    random_state = int(args["--random-state"])

    src_files = sorted(glob(join(in_dir, "*.wav")))
    raw_files = sorted(glob(join(in_dir, "*.raw")))
    is_raw = len(src_files) == 0 and len(raw_files) > 0
    if is_raw:
        print("Assuming 24kHz /16bit audio data")
        src_files = raw_files
    if len(src_files) == 0:
        raise RuntimeError("No files found in {}".format(in_dir))

    total_samples = 0
    indices = []
    signed_int16_max = 2**15

    os.makedirs(out_dir, exist_ok=True)
    if train_dev_test_split:
        os.makedirs(join(out_dir, "train_no_dev"), exist_ok=True)
        os.makedirs(join(out_dir, "dev"), exist_ok=True)
        os.makedirs(join(out_dir, "eval"), exist_ok=True)

    print("Total number of utterances: {}".format(len(src_files)))
    for idx, src_file in tqdm(enumerate(src_files)):
        sr, x = read_wav_or_raw(src_file, is_raw)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / signed_int16_max
        total_samples += len(x)
        total_hours = float(total_samples) / sr / 3600.0
        indices.append(idx)

        if total_hours > limit:
            print("Total hours {:.3f} exceeded limit ({} hours).".format(total_hours, limit))
            break
    print("Total number of collected utterances: {}".format(len(indices)))

    if train_dev_test_split:
        from sklearn.model_selection import train_test_split as split
        # Get test and dev set from last
        if test_size > 1 and dev_size > 1:
            test_size = int(test_size)
            dev_size = int(dev_size)
            testdev_size = test_size + dev_size
            train_indices = indices[:-testdev_size]
            dev_indices = indices[-testdev_size:-testdev_size + dev_size]
            test_indices = indices[-test_size:]
        else:
            train_indices, dev_test_indices = split(
                indices, test_size=test_size + dev_size, random_state=random_state)
            dev_indices, test_indices = split(
                dev_test_indices, test_size=test_size / (test_size + dev_size),
                random_state=random_state)
        sets = [
            (sorted(train_indices), join(out_dir, "train_no_dev")),
            (sorted(dev_indices), join(out_dir, "dev")),
            (sorted(test_indices), join(out_dir, "eval")),
        ]
    else:
        sets = [(indices, out_dir)]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    total_samples = {}
    sr = 0
    for indices, d in sets:
        set_name = basename(d)
        total_samples[set_name] = 0
        for idx in tqdm(indices):
            src_file = src_files[idx]
            dst_path = join(d, basename(src_file))
            if target_sr is not None:
                sr, x = read_wav_or_raw(src_file, is_raw)
                is_int16 = x.dtype == np.int16
                if is_int16:
                    x = x.astype(np.float32) / signed_int16_max
                if target_sr is not None and target_sr != sr:
                    x = librosa.resample(x, sr, target_sr)
                    sr = target_sr
                scaler.partial_fit(x.astype(np.float64).reshape(-1, 1))
                if is_int16:
                    x = (x * signed_int16_max).astype(np.int16)
                write_wav_or_raw(dst_path, sr, x, is_raw)
                total_samples[set_name] += len(x)
            else:
                sr, x = read_wav_or_raw(src_file, is_raw)
                is_int16 = x.dtype == np.int16
                if is_int16:
                    x = x.astype(np.float32) / signed_int16_max
                scaler.partial_fit(x.astype(np.float64).reshape(-1, 1))
                total_samples[set_name] += len(x)
                copy2(src_file, dst_path)

    print("Waveform min: {}".format(scaler.data_min_))
    print("Waveform max: {}".format(scaler.data_max_))
    absmax = max(np.abs(scaler.data_min_[0]), np.abs(scaler.data_max_[0]))
    print("Waveform absolute max: {}".format(absmax))
    if absmax > 1.0:
        print("There were clipping(s) in your dataset.")
    print("Global scaling factor would be around {}".format(1.0 / absmax))

    if train_dev_test_split:
        print("Train/dev/test split:")
        for n, s in zip(["train_no_dev", "dev", "eval"], sets):
            hours = total_samples[n] / sr / 3600.0
            print("{}: {:.2f} hours ({} utt)".format(n, hours, len(s[0])))

    sys.exit(0)
