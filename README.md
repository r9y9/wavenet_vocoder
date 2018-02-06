# WaveNet vocoder

[![PyPI](https://img.shields.io/pypi/v/wavenet_vocoder.svg)](https://pypi.python.org/pypi/wavenet_vocoder)
[![Build Status](https://travis-ci.org/r9y9/wavenet_vocoder.svg?branch=master)](https://travis-ci.org/r9y9/wavenet_vocoder)

The goal of the repository is to provide an implementation of the WaveNet vocoder, which can generate high quality raw speech samples conditioned on linguistic or acoustic features.

Audio samples are available at https://r9y9.github.io/wavenet_vocoder/.

See https://github.com/r9y9/wavenet_vocoder/issues/1 for planned TODOs and current progress.


## Highlights

- Focus on local and global conditioning of WaveNet, which is essential for vocoder.
- Mixture of logistic distributions loss / sampling (experimental)

## Requirements

- Python 3
- CUDA >= 8.0
- PyTorch >= v0.3
- TensorFlow >= v1.3

## Installation

The repository contains a core library (PyTorch implementation of the WaveNet) and utility scripts. All the library and its dependencies can be installed by:

```
git clone https://github.com/r9y9/wavenet_vocoder
cd wavenet_vocoder
pip install -e ".[train]"
```

If you only need the library part, then you can install it by the following command:

```
pip install wavenet_vocoder
```


## Getting started

### 0. Download dataset

- CMU ARCTIC (en): http://festvox.org/cmu_arctic/
- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/

### 1. Preprocessing

In this step, we will extract time-aligned audio and mel-spectrogram.

Usage:

```
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir}
```

Supported `${dataset_name}`s for now are

- `cmu_arctic` (multi-speaker)
- `ljspeech` (single speaker)

Suppose you will want to preprocess CMU ARCTIC dataset and have data in `~/data/cmu_arctic`, then you can preprocess data by:

```
python preprocess.py cmu_arctic ~/data/cmu_arctic ./data/cmu_arctic
```

When this is done, you will see time-aligned extracted features (pairs of audio and mel-spectrogram) in `./data/cmu_arctic`.

### 2. Training

Usage:

```
python train.py --data-root=${data-root} --hparams="parameters you want to override"
```

Important options:

- `--speaker-id=<n>`: It specifies which speaker of data we use for training. If this is not specified, all training data are used. This should only be specified when you are dealing with a multi-speaker dataset. For example, if you are trying to build a speaker-dependent WaveNet vocoder for speaker `awb` of CMU ARCTIC, then you have to specify `--speaker-id=0`. Speaker ID is automatically assigned as follows:

```py
In [1]: from nnmnkwii.datasets import cmu_arctic

In [2]: [(i, s) for (i,s) in enumerate(cmu_arctic.available_speakers)]
Out[2]:

[(0, 'awb'),
 (1, 'bdl'),
 (2, 'clb'),
 (3, 'jmk'),
 (4, 'ksp'),
 (5, 'rms'),
 (6, 'slt')]
```

#### Training un-conditional WaveNet

```
python train.py --data-root=./data/cmu_arctic/
    --hparams="cin_channels=-1,gin_channels=-1"
```

You have to disable global and local conditioning by setting `gin_channels` and `cin_channels` to negative values.

#### Training WaveNet conditioned on mel-spectrogram

```
python train.py --data-root=./data/cmu_arctic/ --speaker-id=0 \
    --hparams="cin_channels=80,gin_channels=-1"
```

#### Training WaveNet conditioned on mel-spectrogram and speaker embedding

```
python train.py --data-root=./data/cmu_arctic/ \
    --hparams="cin_channels=80,gin_channels=16,n_speakers=7"
```

### 3. Monitor with Tensorboard

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```

### 4. Synthesize from a checkpoint

Usage:

```
python synthesis.py ${checkpoint_path} ${output_dir} -hparams="parameters you want to override"
```

Important options:

- `--length=<n>`: Number of time steps to generate. This is only valid for un-conditional WaveNets.
- `--conditional=<path>`: Path of local conditional features (.npy). If this is specified, number of time steps to generate is determined by the size of conditional feature.

e.g.,

```
python synthesis.py --hparams="parameters you want to override" \ 
    checkpoints_awb/checkpoint_step000100000.pth \
    generated/test_awb \
    --conditional=./data/cmu_arctic/cmu_arctic-mel-00001.npy
```

## Misc

### Synthesize audio samples for testset

Usage:


```
python evaluate.py ${checkpoint_path} ${output_dir} --data-root="data location"\
    --hparams="parameters you want to override"
```

Options:

- `--data-root`: Data root. This is required to collect testset.
- `--num-utterances`: (For multi-speaker model) number of utterances to be generated per speaker. This is useful especially when testset is large and don't want to generate all utterances. For single speaker dataset, you can hit `ctrl-c` whenever you want to stop evaluation.

e.g.,

```
python evaluate.py --data-root=./data/cmu_arctic/ \
    ./checkpoints_awb/checkpoint_step000100000.pth \
    ./generated/cmu_arctic_awb
```

## References

- [Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", 	arXiv:1609.03499, Sep 2016.](https://arxiv.org/abs/1609.03499)
- [Aaron van den Oord, Yazhe Li, Igor Babuschkin, et al, "Parallel WaveNet: Fast High-Fidelity Speech Synthesis", 	arXiv:1711.10433, Nov 2017.](https://arxiv.org/abs/1711.10433)
- [Tamamori, Akira, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0314.PDF)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
- [Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.](https://arxiv.org/abs/1710.07654)
