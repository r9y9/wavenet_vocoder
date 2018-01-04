# coding: utf-8
"""
Synthesis waveform for testset

usage: evaluate.py [options] <checkpoint> <dst_dir>

options:
    --data-root=<dir>           Directory contains preprocessed features.
    --hparams=<parmas>          Hyper parameters [default: ].
    --length=<T>                Steps to generate [default: 32000].
    --speaker-id=<N>            Use specific speaker of data in case for multi-speaker datasets.
    --initial-value=<n>         Initial value for the WaveNet decoder.
    --file-name-suffix=<s>      File name suffix [default: ].
    --output-html               Output html for blog post.
    -h, --help                  Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
from torch.autograd import Variable
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa

import audio
from hparams import hparams


use_cuda = torch.cuda.is_available()


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "cmu_arctic")
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]

    length = int(args["--length"])
    # Note that speaker-id is used for filtering out unrelated-speaker from
    # multi-speaker dataset.
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else int(initial_value)
    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    from train import build_model, get_data_loaders
    from synthesis import wavegen

    # Data
    # Use exactly same testset used in training script
    # disable shuffle for convenience
    test_data_loader = get_data_loaders(data_root, speaker_id, test_shuffle=False)["test"]
    test_dataset = test_data_loader.dataset

    # Model
    model = build_model()

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    os.makedirs(dst_dir, exist_ok=True)
    dst_dir_name = basename(os.path.normpath(dst_dir))

    for idx, (x, c, g) in enumerate(test_dataset):
        target_audio_path = test_dataset.X.collected_files[idx][0]
        if output_html:
            def _tqdm(x): return x
        else:
            _tqdm = tqdm
            print("Target audio is {}".format(target_audio_path))
            if c is not None:
                print("Conditioned by {}".format(test_dataset.Mel.collected_files[idx][0]))

        # Paths
        dst_wav_path = join(dst_dir, "{}_{}{}_predicted.wav".format(
            idx, checkpoint_name, file_name_suffix))
        target_wav_path = join(dst_dir, "{}_{}{}_target.wav".format(
            idx, checkpoint_name, file_name_suffix))

        # Generate
        waveform = wavegen(model, length, c=c, g=g, initial_value=initial_value,
                           fast=True, tqdm=_tqdm)

        # save
        librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)
        librosa.output.write_wav(target_wav_path, P.inv_mulaw_quantize(x),
                                 sr=hparams.sample_rate)

        # log
        if output_html:
            print("""
<audio controls="controls" >
<source src="/{}/audio/{}/{}" autoplay/>
Your browser does not support the audio element.
</audio>
""".format(hparams.name, dst_dir_name, basename(dst_wav_path)))

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
