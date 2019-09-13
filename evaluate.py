# coding: utf-8
"""
Synthesis waveform for testset

usage: evaluate.py [options] <dump-root> <checkpoint> <dst_dir>

options:
    --hparams=<parmas>          Hyper parameters [default: ].
    --preset=<json>             Path of preset parameters (json).
    --length=<T>                Steps to generate [default: 32000].
    --speaker-id=<N>            Use specific speaker of data in case for multi-speaker datasets.
    --initial-value=<n>         Initial value for the WaveNet decoder.
    --output-html               Output html for blog post.
    --num-utterances=N>         Generate N utterenaces per speaker [default: -1].
    --verbose=<level>           Verbosity level [default: 0].
    -h, --help                  Show help message.
"""
from docopt import docopt

import sys
from glob import glob
import os
from os.path import dirname, join, basename, splitext, exists
import torch
import numpy as np
from nnmnkwii import preprocessing as P
from tqdm import tqdm
from scipy.io import wavfile
from torch.utils import data as data_utils
from torch.nn import functional as F

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams
from train import RawAudioDataSource, MelSpecDataSource, PyTorchDataset, _pad_2d
from nnmnkwii.datasets import FileSourceDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def to_int16(x):
    if x.dtype == np.int16:
        return x
    assert x.dtype == np.float32
    assert x.min() >= -1 and x.max() <= 1.0
    return (x * 32767).astype(np.int16)


def dummy_collate(batch):
    N = len(batch)
    input_lengths = [(len(x) - hparams.cin_pad * 2) * audio.get_hop_size() for x in batch]
    input_lengths = torch.LongTensor(input_lengths)
    max_len = max([len(x) for x in batch])
    c_batch = np.array([_pad_2d(x, max_len) for x in batch], dtype=np.float32)
    c_batch = torch.FloatTensor(c_batch).transpose(1, 2).contiguous()
    return [None]*N, [None]*N, c_batch, None, input_lengths


def get_data_loader(data_dir, collate_fn):
    wav_paths = glob(join(data_dir, "*-wave.npy"))
    if len(wav_paths) != 0:
        X = FileSourceDataset(RawAudioDataSource(data_dir,
                                                 hop_size=audio.get_hop_size(),
                                                 max_steps=None, cin_pad=hparams.cin_pad))
    else:
        X = None
    C = FileSourceDataset(MelSpecDataSource(data_dir,
                                            hop_size=audio.get_hop_size(),
                                            max_steps=None, cin_pad=hparams.cin_pad))
    # No audio found:
    if X is None:
        assert len(C) > 0
        data_loader = data_utils.DataLoader(
            C, batch_size=hparams.batch_size, drop_last=False,
            num_workers=hparams.num_workers, sampler=None, shuffle=False,
            collate_fn=dummy_collate, pin_memory=hparams.pin_memory)
    else:
        assert len(X) == len(C)
        if C[0].shape[-1] != hparams.cin_channels:
            raise RuntimeError(
                """Invalid cin_channnels {}. Expectd to be {}.""".format(
                    hparams.cin_channels, C[0].shape[-1]))
        dataset = PyTorchDataset(X, C)

        data_loader = data_utils.DataLoader(
            dataset, batch_size=hparams.batch_size, drop_last=False,
            num_workers=hparams.num_workers, sampler=None, shuffle=False,
            collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    return data_loader


if __name__ == "__main__":
    args = docopt(__doc__)
    verbose = int(args["--verbose"])
    if verbose > 0:
        print("Command line args:\n", args)
    data_root = args["<dump-root>"]
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]

    length = int(args["--length"])
    # Note that speaker-id is used for filtering out unrelated-speaker from
    # multi-speaker dataset.
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else float(initial_value)
    output_html = args["--output-html"]
    num_utterances = int(args["--num-utterances"])
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    else:
        hparams_json = join(dirname(checkpoint_path), "hparams.json")
        if exists(hparams_json):
            print("Loading hparams from {}".format(hparams_json))
            with open(hparams_json) as f:
                hparams.parse_json(f.read())

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    hparams.max_time_sec = None
    hparams.max_time_steps = None

    from train import build_model, get_data_loaders
    from synthesis import batch_wavegen

    # Data
    # Use exactly same testset used in training script
    # disable shuffle for convenience
    # test_data_loader = get_data_loaders(data_root, speaker_id, test_shuffle=False)["test"]
    from train import collate_fn
    test_data_loader = get_data_loader(data_root, collate_fn)
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
    cin_pad = hparams.cin_pad
    file_idx = 0
    for idx, (x, y, c, g, input_lengths) in enumerate(test_data_loader):
        if cin_pad > 0:
            c = F.pad(c, pad=(cin_pad, cin_pad), mode="replicate")

        # B x 1 x T
        if x[0] is not None:
            B, _, T = x.shape
        else:
            B, _, Tn = c.shape
            T = Tn * audio.get_hop_size()

        if g is None and num_utterances > 0 and B * idx >= num_utterances:
            break

        ref_files = []
        ref_feats = []
        for i in range(B):
            # Yes this is ugly...
            if hasattr(test_data_loader.dataset, "X"):
                ref_files.append(test_data_loader.dataset.X.collected_files[file_idx][0])
            else:
                pass
            if hasattr(test_data_loader.dataset, "Mel"):
                ref_feats.append(test_data_loader.dataset.Mel.collected_files[file_idx][0])
            else:
                ref_feats.append(test_data_loader.dataset.collected_files[file_idx][0])
            file_idx += 1

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

        # Generate
        y_hats = batch_wavegen(model, c=c, g=g, fast=True, tqdm=_tqdm)

        # Save each utt.
        has_ref_file = len(ref_files) > 0
        for i, (ref, gen, length) in enumerate(zip(x, y_hats, input_lengths)):
            if has_ref_file:
                if is_mulaw_quantize(hparams.input_type):
                    # needs to be float since mulaw_inv returns in range of [-1, 1]
                    ref = ref.max(0)[1].view(-1).float().cpu().numpy()[:length]
                else:
                    ref = ref.view(-1).cpu().numpy()[:length]
            gen = gen[:length]
            if has_ref_file:
                target_audio_path = ref_files[i]
                name = splitext(basename(target_audio_path))[0].replace("-wave", "")
            else:
                target_feat_path = ref_feats[i]
                name = splitext(basename(target_feat_path))[0].replace("-feats", "")

            # Paths
            if g is None:
                dst_wav_path = join(dst_dir, "{}_gen.wav".format(
                    name))
                target_wav_path = join(dst_dir, "{}_ref.wav".format(
                    name))
            else:
                dst_wav_path = join(dst_dir, "speaker{}_{}_gen.wav".format(
                    g, name))
                target_wav_path = join(dst_dir, "speaker{}_{}_ref.wav".format(
                    g, name))

            # save
            if has_ref_file:
                if is_mulaw_quantize(hparams.input_type):
                    ref = P.inv_mulaw_quantize(ref, hparams.quantize_channels - 1)
                elif is_mulaw(hparams.input_type):
                    ref = P.inv_mulaw(ref, hparams.quantize_channels - 1)
                if hparams.postprocess is not None and hparams.postprocess not in ["", "none"]:
                    ref = getattr(audio, hparams.postprocess)(ref)
                if hparams.global_gain_scale > 0:
                    ref /= hparams.global_gain_scale

            # clip (just in case)
            gen = np.clip(gen, -1.0, 1.0)
            if has_ref_file:
                ref = np.clip(ref, -1.0, 1.0)

            wavfile.write(dst_wav_path, hparams.sample_rate, to_int16(gen))
            if has_ref_file:
                wavfile.write(target_wav_path, hparams.sample_rate, to_int16(ref))

            # log (TODO)
            if output_html and False:
                print("""
    <audio controls="controls" >
    <source src="/{}/audio/{}/{}" autoplay/>
    Your browser does not support the audio element.
    </audio>
    """.format(hparams.name, dst_dir_name, basename(dst_wav_path)))

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
