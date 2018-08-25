# coding: utf-8
"""
Synthesis waveform for testset

usage: evaluate.py [options] <checkpoint> <dst_dir>

options:
    --data-root=<dir>           Directory contains preprocessed features.
    --hparams=<parmas>          Hyper parameters [default: ].
    --preset=<json>             Path of preset parameters (json).
    --length=<T>                Steps to generate [default: 32000].
    --speaker-id=<N>            Use specific speaker of data in case for multi-speaker datasets.
    --initial-value=<n>         Initial value for the WaveNet decoder.
    --file-name-suffix=<s>      File name suffix [default: ].
    --output-html               Output html for blog post.
    --num-utterances=N>         Generate N utterenaces per speaker [default: -1].
    -h, --help                  Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa


from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
    initial_value = None if initial_value is None else float(initial_value)
    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]
    num_utterances = int(args["--num-utterances"])
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
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
    model = build_model().to(device)

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    os.makedirs(dst_dir, exist_ok=True)
    dst_dir_name = basename(os.path.normpath(dst_dir))

    generated_utterances = {}
    for idx, (x, c, g) in enumerate(test_dataset):
        target_audio_path = test_dataset.X.collected_files[idx][0]
        if g is None and num_utterances > 0 and idx > num_utterances:
            break
        if num_utterances > 0 and g is not None:
            try:
                generated_utterances[g] += 1
                if generated_utterances[g] > num_utterances:
                    continue
            except KeyError:
                generated_utterances[g] = 1

        if output_html:
            def _tqdm(x): return x
        else:
            _tqdm = tqdm
            print("Target audio is {}".format(target_audio_path))
            if c is not None:
                print("Local conditioned by {}".format(test_dataset.Mel.collected_files[idx][0]))
            if g is not None:
                print("Global conditioned by speaker id {}".format(g))

        # Paths
        if g is None:
            dst_wav_path = join(dst_dir, "{}_{}{}_predicted.wav".format(
                idx, checkpoint_name, file_name_suffix))
            target_wav_path = join(dst_dir, "{}_{}{}_target.wav".format(
                idx, checkpoint_name, file_name_suffix))
        else:
            dst_wav_path = join(dst_dir, "speaker{}_{}_{}{}_predicted.wav".format(
                g, idx, checkpoint_name, file_name_suffix))
            target_wav_path = join(dst_dir, "speaker{}_{}_{}{}_target.wav".format(
                g, idx, checkpoint_name, file_name_suffix))

        # Generate
        waveform = wavegen(model, length, c=c, g=g, initial_value=initial_value,
                           fast=True, tqdm=_tqdm)

        # save
        librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)
        if is_mulaw_quantize(hparams.input_type):
            x = P.inv_mulaw_quantize(x, hparams.quantize_channels)
        elif is_mulaw(hparams.input_type):
            x = P.inv_mulaw(x, hparams.quantize_channels)
        librosa.output.write_wav(target_wav_path, x, sr=hparams.sample_rate)

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
