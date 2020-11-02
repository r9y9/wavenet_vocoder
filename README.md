# WaveNet vocoder

[![PyPI](https://img.shields.io/pypi/v/wavenet_vocoder.svg)](https://pypi.python.org/pypi/wavenet_vocoder)
[![Build Status](https://travis-ci.org/r9y9/wavenet_vocoder.svg?branch=master)](https://travis-ci.org/r9y9/wavenet_vocoder)
[![Build status](https://ci.appveyor.com/api/projects/status/lvt9jtimtg0koxwj?svg=true)](https://ci.appveyor.com/project/r9y9/wavenet-vocoder)
[![DOI](https://zenodo.org/badge/115492234.svg)](https://zenodo.org/badge/latestdoi/115492234)

**NOTE**: This is the development version. If you need a stable version, please checkout the v0.1.1.

The goal of the repository is to provide an implementation of the WaveNet vocoder, which can generate high quality raw speech samples conditioned on linguistic or acoustic features.

Audio samples are available at https://r9y9.github.io/wavenet_vocoder/.

## News

- 2019/10/31: The repository has been adapted to [ESPnet](https://github.com/espnet/espnet). English, Chinese, and Japanese samples and pretrained models are available there. See https://github.com/espnet/espnet and https://github.com/espnet/espnet#tts-results for details.

## Online TTS demo

A notebook supposed to be executed on https://colab.research.google.com is available:

- [Tacotron2: WaveNet-based text-to-speech demo](https://colab.research.google.com/github/r9y9/Colaboratory/blob/master/Tacotron2_and_WaveNet_text_to_speech_demo.ipynb)

## Highlights

- Focus on local and global conditioning of WaveNet, which is essential for vocoder.
- 16-bit raw audio modeling by mixture distributions: mixture of logistics (MoL), mixture of Gaussians, and single Gaussian distributions are supported.
- Various audio samples and pre-trained models
- Fast inference by caching intermediate states in convolutions. Similar to [arXiv:1611.09482](https://arxiv.org/abs/1611.09482)
- Integration with ESPNet (https://github.com/espnet/espnet)

## Pre-trained models

**Note**: This is not itself a text-to-speech (TTS) model. With a pre-trained model provided here, you can synthesize waveform given a *mel spectrogram*, not raw text. You will need mel-spectrogram prediction model (such as Tacotron2) to use the pre-trained models for TTS.

**Note**: As for the pretrained model for LJSpeech, the model was fine-tuned multiple times and trained for more than 1000k steps in total. Please refer to the issues ([#1](https://github.com/r9y9/wavenet_vocoder/issues/1#issuecomment-361130247), [#75](https://github.com/r9y9/wavenet_vocoder/issues/75), [#45](https://github.com/r9y9/wavenet_vocoder/issues/45#issuecomment-383313651)) to know how the model was trained.

| Model URL                                                                                                                        | Data       | Hyper params URL                                                                                     | Git commit                                                                                         | Steps         |
|----------------------------------------------------------------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|---------------|
| [link](https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth?dl=0)                      | LJSpeech   | [link](https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json?dl=0)                | [2092a64](https://github.com/r9y9/wavenet_vocoder/commit/2092a647e60ce002389818de1fa66d0a2c5763d8) | 1000k~  steps |
| [link](https://www.dropbox.com/s/d0qk4ow9uuh2lww/20180212_mixture_multispeaker_cmu_arctic_checkpoint_step000740000_ema.pth?dl=0) | CMU ARCTIC | [link](https://www.dropbox.com/s/i35yigj5hvmeol8/20180212_multispeaker_cmu_arctic_mixture.json?dl=0) | [b1a1076](https://github.com/r9y9/wavenet_vocoder/tree/b1a1076e8b5d9b3e275c28f2f7f4d7cd0e75dae4)   | 740k steps    |

To use pre-trained models, first checkout the specific git commit noted above. i.e.,

```
git checkout ${commit_hash}
```

And then follows "Synthesize from a checkpoint" section in the README. Note that old version of synthesis.py may not accept `--preset=<json>` parameter and you might have to change `hparams.py` according to the preset (json) file.

You could try for example:

```
# Assuming you have downloaded LJSpeech-1.1 at ~/data/LJSpeech-1.1
# pretrained model (20180510_mixture_lj_checkpoint_step000320000_ema.pth)
# hparams (20180510_mixture_lj_checkpoint_step000320000_ema.json)
git checkout 2092a64
python preprocess.py ljspeech ~/data/LJSpeech-1.1 ./data/ljspeech \
  --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json
python synthesis.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json \
  --conditional=./data/ljspeech/ljspeech-mel-00001.npy \
  20180510_mixture_lj_checkpoint_step000320000_ema.pth \
  generated
```

You can find a generated wav file in `generated` directory. Wonder how it works? then take a look at code:)

## Repository structure

The repository consists of 1) pytorch library, 2) command line tools, and 3) [ESPnet](https://github.com/espnet/espnet)-style recipes. The first one is a pytorch library to provide WavaNet functionality. The second one is a set of tools to run WaveNet training/inference, data processing, etc. The last one is the reproducible recipes combining the WaveNet library and utility tools. Please take a look at them depending on your purpose. If you want to build your WaveNet on your dataset (I guess this is the most likely case), the recipe is the way for you.

## Requirements

- Python 3
- CUDA >= 8.0
- PyTorch >= v0.4.0

## Installation


```
git clone https://github.com/r9y9/wavenet_vocoder && cd wavenet_vocoder
pip install -e .
```

If you only need the library part, you can install it from pypi:

```
pip install wavenet_vocoder
```

## Getting started

### Kaldi-style recipes

The repository provides Kaldi-style recipes to make experiments reproducible and easily manageable. Available recipes are as follows:

- `mulaw256`: WaveNet that uses categorical output distribution. The input is 8-bit mulaw quantized waveform.
- `mol`: Mixture of Logistics (MoL) WaveNet. The input is 16-bit raw audio.
- `gaussian`: Single-Gaussian WaveNet (a.k.a. teacher WaveNet of [ClariNet](https://clarinet-demo.github.io/)). The input is 16-bit raw audio.

All the recipe has `run.sh`, which specifies all the steps to perform WaveNet training/inference including data preprocessing. Please see run.sh in [egs](egs) directory for details.

**NOTICE**: Global conditioning for multi-speaker WaveNet is not supported in the above recipes (it shouldn't be difficult to implement though). Please check v0.1.12 for the feature, or if you *really* need the feature, please raise an issue.

#### Apply recipe to your own dataset

The recipes are designed to be generic so that one can use them for any dataset. To apply recipes to your own dataset, you'd need to put *all* the wav files in a single flat directory. i.e.,

```
> tree -L 1 ~/data/LJSpeech-1.1/wavs/ | head
/Users/ryuichi/data/LJSpeech-1.1/wavs/
├── LJ001-0001.wav
├── LJ001-0002.wav
├── LJ001-0003.wav
├── LJ001-0004.wav
├── LJ001-0005.wav
├── LJ001-0006.wav
├── LJ001-0007.wav
├── LJ001-0008.wav
├── LJ001-0009.wav
```

That's it! The last step is to modify `db_root` in run.sh or give `db_root` as the command line argment for run.sh.

```
./run.sh --stage 0 --stop-stage 0 --db-root ~/data/LJSpeech-1.1/wavs/
```

### Step-by-step

A recipe typically consists of multiple steps. It is strongly recommended to run the recipe step-by-step to understand how it works for the first time. To do so, specify `stage` and `stop_stage` as follows:

```
./run.sh --stage 0 --stop-stage 0
```

```
./run.sh --stage 1 --stop-stage 1
```

```
./run.sh --stage 2 --stop-stage 2
```

In typical situations, you'd need to specify CUDA devices explciitly expecially for training step.

```
CUDA_VISIBLE_DEVICES="0,1" ./run.sh --stage 2 --stop-stage 2
```

### Docs for command line tools

Command line tools are writtern with [docopt](http://docopt.org/). See each docstring for the basic usages.

#### tojson.py

Dump hyperparameters to a json file.

Usage:

```
python tojson.py --hparams="parameters you want to override" <output_json_path>
```

#### preprocess.py

Usage:

```
python preprocess.py wavallin ${dataset_path} ${out_dir} --preset=<json>
```

#### train.py

> Note: for multi gpu training, you have better ensure that batch_size % num_gpu == 0

Usage:

```
python train.py --dump-root=${dump-root} --preset=<json>\
  --hparams="parameters you want to override"
```


#### evaluate.py

Given a directoy that contains local conditioning features, synthesize waveforms for them.

Usage:

```
python evaluate.py ${dump_root} ${checkpoint} ${output_dir} --dump-root="data location"\
    --preset=<json> --hparams="parameters you want to override"
```

Options:

- `--num-utterances=<N>`: Number of utterances to be generated. If not specified, generate all uttereances. This is useful for debugging.

#### synthesis.py

**NOTICE**: This is probably not working now. Please use evaluate.py instead.

Synthesize waveform give a conditioning feature.

Usage:

```
python synthesis.py ${checkpoint_path} ${output_dir} --preset=<json> --hparams="parameters you want to override"
```

Important options:

- `--conditional=<path>`: (Required for conditional WaveNet) Path of local conditional features (.npy). If this is specified, number of time steps to generate is determined by the size of conditional feature.


### Training scenarios

#### Training un-conditional WaveNet

**NOTICE**: This is probably not working now. Please check v0.1.1 for the working version.

```
python train.py --dump-root=./data/cmu_arctic/
    --hparams="cin_channels=-1,gin_channels=-1"
```

You have to disable global and local conditioning by setting `gin_channels` and `cin_channels` to negative values.

#### Training WaveNet conditioned on mel-spectrogram

```
python train.py --dump-root=./data/cmu_arctic/ --speaker-id=0 \
    --hparams="cin_channels=80,gin_channels=-1"
```

#### Training WaveNet conditioned on mel-spectrogram and speaker embedding

**NOTICE**: This is probably not working now. Please check v0.1.1 for the working version.

```
python train.py --dump-root=./data/cmu_arctic/ \
    --hparams="cin_channels=80,gin_channels=16,n_speakers=7"
```

### Misc

#### Monitor with Tensorboard

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```


### List of papers that used the repository

- A Comparison of Recent Neural Vocoders for Speech Signal Reconstruction https://www.isca-speech.org/archive/SSW_2019/abstracts/SSW10_O_1-2.html
- WaveGlow: A Flow-based Generative Network for Speech Synthesis https://arxiv.org/abs/1811.00002
- WaveCycleGAN2: Time-domain Neural Post-filter for Speech Waveform Generation https://arxiv.org/abs/1904.02892
- Parametric Resynthesis with neural vocoders https://arxiv.org/abs/1906.06762
- Representation Mixing fo TTS Synthesis https://arxiv.org/abs/1811.07240
- A Unified Neural Architecture for Instrumental Audio Tasks https://arxiv.org/abs/1903.00142
- ESPnet-TTS: Unified, Reproducible, and Integratable Open Source End-to-End Text-to-Speech Toolkit: https://arxiv.org/abs/1910.10909

Thank you very much!! If you find a new one, please submit a PR.

## Sponsors

- https://github.com/echelon

## References

- [Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", 	arXiv:1609.03499, Sep 2016.](https://arxiv.org/abs/1609.03499)
- [Aaron van den Oord, Yazhe Li, Igor Babuschkin, et al, "Parallel WaveNet: Fast High-Fidelity Speech Synthesis", 	arXiv:1711.10433, Nov 2017.](https://arxiv.org/abs/1711.10433)
- [Tamamori, Akira, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0314.PDF)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
- [Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.](https://arxiv.org/abs/1710.07654)
- [Tom Le Paine, Pooya Khorrami, Shiyu Chang, et al, "Fast Wavenet Generation Algorithm", arXiv:1611.09482, Nov. 2016](https://arxiv.org/abs/1611.09482)
- [Ye Jia, Yu Zhang, Ron J. Weiss, Quan Wang, Jonathan Shen, Fei Ren, Zhifeng Chen, Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu, et al, "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis" , arXiv:1806.04558v4 cs.CL 2 Jan 2019](https://arxiv.org/abs/1806.04558)
