# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from os.path import join
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams


def preprocess(mod, in_dir, out_root, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    sr = hparams.sample_rate
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Min frame length: %d' % min(m[2] for m in metadata))
    print('Max frame length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() // 2 if num_workers is None else int(num_workers)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    print("Sampling frequency: {}".format(hparams.sample_rate))
    if name in ["cmu_arctic", "jsut", "librivox"]:
        print("""warn!: {} is no longer explicitly supported!

Please use a generic dataest 'wavallin' instead.
All you need to do is to put all wav files in a single directory.""".format(name))
        sys.exit(1)

    if name == "ljspeech":
        print("""warn: ljspeech is deprecated!
Please use a generic dataset 'wavallin' instead.""")
        sys.exit(1)

    mod = importlib.import_module("datasets." + name)
    preprocess(mod, in_dir, out_dir, num_workers)
