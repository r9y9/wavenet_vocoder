from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio

from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    # with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
    #    for line in f:
    #        parts = line.strip().split('|')
    #        wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
    #        text = parts[2]
    #        futures.append(executor.submit(
    #            partial(_process_utterance, out_dir, index, wav_path, text)))
    #        index += 1

    valid_ext = '.ogg .wav .mp3'.split()
    for f in sorted(os.listdir(in_dir)):
        valid = sum([f.endswith(ext) for ext in valid_ext])
        if valid < 1:
            continue

        audio_filepath = os.path.join(in_dir, f)
        text = audio_filepath  # Not very informative
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index, audio_filepath, text)))
        index += 1
    return [tup for future in tqdm(futures) for tup in future.result()]


def _process_utterance(out_dir, index, audio_filepath, text):
    # Load the audio to a numpy array:
    wav_whole = audio.load_wav(audio_filepath)

    if hparams.rescaling:
        wav_whole = wav_whole / np.abs(wav_whole).max() * hparams.rescaling_max

    # This is a librivox source, so the audio files are going to be v. long
    # compared to a typical 'utterance' : So split the wav into chunks

    tup_results = []

    n_samples = int(8.0 * hparams.sample_rate)  # All 8 second utterances
    n_chunks = wav_whole.shape[0] // n_samples

    for chunk_idx in range(n_chunks):
        chunk_start, chunk_end = chunk_idx * n_samples, (chunk_idx + 1) * n_samples
        if chunk_idx == n_chunks - 1:  # This is the last chunk - allow it to extend to the end of the file
            chunk_end = None
        wav = wav_whole[chunk_start: chunk_end]

        # Mu-law quantize
        if is_mulaw_quantize(hparams.input_type):
            # [0, quantize_channels)
            out = P.mulaw_quantize(wav, hparams.quantize_channels)

            # Trim silences
            start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
            wav = wav[start:end]
            out = out[start:end]
            constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
            out_dtype = np.int16
        elif is_mulaw(hparams.input_type):
            # [-1, 1]
            out = P.mulaw(wav, hparams.quantize_channels)
            constant_values = P.mulaw(0.0, hparams.quantize_channels)
            out_dtype = np.float32
        else:
            # [-1, 1]
            out = wav
            constant_values = 0.0
            out_dtype = np.float32

        # Compute a mel-scale spectrogram from the trimmed wav:
        # (N, D)
        mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
        # lws pads zeros internally before performing stft
        # this is needed to adjust time resolution between audio and mel-spectrogram
        l, r = audio.lws_pad_lr(wav, hparams.fft_size, audio.get_hop_size())

        # zero pad for quantized signal
        out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
        N = mel_spectrogram.shape[0]
        assert len(out) >= N * audio.get_hop_size()

        # time resolution adjustment
        # ensure length of raw audio is multiple of hop_size so that we can use
        # transposed convolution to upsample
        out = out[:N * audio.get_hop_size()]
        assert len(out) % audio.get_hop_size() == 0

        timesteps = len(out)

        # Write the spectrograms to disk:
        audio_filename = 'librivox-audio-%04d-%05d.npy' % (index, chunk_idx,)
        mel_filename = 'librivox-mel-%04d-%05d.npy' % (index, chunk_idx,)
        text_idx = '%s - %05d' % (text, chunk_idx,)
        np.save(os.path.join(out_dir, audio_filename),
                out.astype(out_dtype), allow_pickle=False)
        np.save(os.path.join(out_dir, mel_filename),
                mel_spectrogram.astype(np.float32), allow_pickle=False)

        # Add results tuple describing this training example:
        tup_results.append((audio_filename, mel_filename, timesteps, text_idx))

    # Return all the audio results tuples (unpack in caller)
    return tup_results
