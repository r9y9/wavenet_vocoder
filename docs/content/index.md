+++
Categories = []
Description = ""
Keywords = []
Tags = []
date = "2018-01-04T19:42:01+09:00"
title = "index"
type = "index"
+++

<br/>

- Github: https://github.com/r9y9/wavenet_vocoder

This page provides audio samples for the open source implementation of the **WaveNet (WN)** vocoder.
Text-to-speech samples are found at the last section.

- WN conditioned on mel-spectrogram (16-bit linear PCM, 22.5kHz)
- WN conditioned on mel-spectrogram (8-bit mu-law, 16kHz)
- WN conditioned on mel-spectrogram and speaker-embedding (16-bit linear PCM, 16kHz)
- Tacotron2: WN-based text-to-speech (**New!**)

## WN conditioned on mel-spectrogram (16-bit linear PCM, 22.5kHz)

- Samples from a model trained for over 400k steps.
- Left: generated, Right: ground truth

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/0_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/0_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/1_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/1_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/2_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/2_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/3_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/3_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/4_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/4_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/5_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/5_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/6_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/6_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/7_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/7_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/8_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/8_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/9_checkpoint_step000410000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/mixture_lj/9_checkpoint_step000410000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

| key                         | value |
|---------------------------------|------------------------------------------------------|
| Data                            | LJSpeech (12522 for training, 578 for testing) |
| Input type | 16-bit linear PCM |
| Sampling frequency  | 22.5kHz |
| Local conditioning            | 80-dim mel-spectrogram                               |
| Hop size | 256 |
| Global conditioning            | N/A                              |
| Total layers                    | 24                                                   |
| Num cycles                      | 4                                                   |
| Residual / Gate / Skip-out channels | 512 / 512 / 256  |
| Receptive field (samples / ms) | 505 / 22.9                                        |
| Numer of mixtures  |  10  |
| Number of upsampling layers | 4 |

## WN conditioned on mel-spectrogram (8-bit mu-law, 16kHz)

- Samples from a model trained for 100k steps (~22 hours)
- Left: generated, Right: (mu-law encoded) ground truth

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/0_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/0_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/1_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/1_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/2_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/2_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/3_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/3_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/4_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/4_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/5_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/5_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/6_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/6_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/7_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/7_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/8_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/8_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/9_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/9_checkpoint_step000100000_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

| key                         | value |
|---------------------------------|------------------------------------------------------|
| Data                            | CMU ARCTIC (`clb`) (1183 for training, 50 for testing) |
| Input type | 8-bit mu-law encoded one-hot vector |
| Sampling frequency  | 16kHz |
| Local conditioning            | 80-dim mel-spectrogram                               |
| Hop size | 256 |
| Global conditioning            | N/A                              |
| Total layers                    | 16                                                   |
| Num cycles                      | 2                                                    |
| Residual / Gate / Skip-out channels | 512 / 512 / 256  |
| Receptive field (samples / ms) | 1021 / 63.8                                          |
| Number of upsampling layers | N/A |


## WN conditioned on mel-spectrogram and speaker-embedding (16-bit linear PCM, 16kHz)

- Samples from a model trained for over 1000k steps
- Left: generated, Right: ground truth

**awb**

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker0_12_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker0_12_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker0_7_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker0_7_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

**bdl**

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker1_2_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker1_2_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker1_33_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker1_33_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

**clb**

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker2_5_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker2_5_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker2_9_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker2_9_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

**jmk**

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker3_24_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker3_24_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker3_30_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker3_30_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>


**ksp**

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker4_25_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker4_25_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker4_3_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker4_3_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>


**rms**

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker5_0_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker5_0_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker5_1_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker5_1_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

**slt**

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker6_4_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker6_4_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker6_6_checkpoint_step000740000_ema_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>
<audio controls="controls" >
<source src="/wavenet_vocoder/audio/cmu_arctic_multispeaker/speaker6_6_checkpoint_step000740000_ema_target.wav" autoplay/>
Your browser does not support the audio element.
</audio>

| key                         | value |
|---------------------------------|------------------------------------------------------|
| Data                            | CMU ARCTIC (7580 for training, 350 for testing) |
| Input type | 16-bit linear PCM |
| Local conditioning            | 80-dim mel-spectrogram                               |
| Hop size | 256 |
| Global conditioning            | 16-dim speaker embedding [^1]                              |
| Total layers                    | 24                                                   |
| Num cycles                      | 4                                                   |
| Residual / Gate / Skip-out channels | 512 / 512 / 256  |
| Receptive field (samples / ms) | 505 / 22.9                                        |
| Numer of mixtures  |  10  |
| Number of upsampling layers | 4 |

[^1]: Note that mel-spectrogram used in local conditioning is dependent on speaker characteristics, so we cannot simply change the speaker identity of the generated audio samples using the model. It should work without speaker embedding, but it might have helped training speed.

## Tacotron2: WN-based text-to-speech

- Tacotron2 (mel-spectrogram prediction part): trained 189k steps on LJSpeech dataset ([Pre-trained model](https://www.dropbox.com/s/vx7y4qqs732sqgg/pretrained.tar.gz?dl=0), [Hyper params](https://github.com/r9y9/Tacotron-2/blob/9ce1a0e65b9217cdc19599c192c5cd68b4cece5b/hparams.py)). The work has been done by [@Rayhane-mamah](https://github.com/Rayhane-mamah). See https://github.com/Rayhane-mamah/Tacotron-2 for details.
- WaveNet: trained over 1000k steps on LJSpeech dataset ([Pre-trained model](https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth?dl=0), [Hyper params](https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json?dl=0))


Scientists at the CERN laboratory say they have discovered a new particle.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00001.wav" autoplay/>
Your browser does not support the audio element.
</audio>


There's a way to measure the acute emotional intelligence that has never gone out of style.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00002.wav" autoplay/>
Your browser does not support the audio element.
</audio>


President Trump met with other leaders at the Group of 20 conference.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00003.wav" autoplay/>
Your browser does not support the audio element.
</audio>


The Senate's bill to repeal and replace the Affordable Care Act is now imperiled.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00004.wav" autoplay/>
Your browser does not support the audio element.
</audio>


Generative adversarial network or variational auto-encoder.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00005.wav" autoplay/>
Your browser does not support the audio element.
</audio>


Basilar membrane and otolaryngology are not auto-correlations.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00006.wav" autoplay/>
Your browser does not support the audio element.
</audio>


He has read the whole thing.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00007.wav" autoplay/>
Your browser does not support the audio element.
</audio>


He reads books.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00008.wav" autoplay/>
Your browser does not support the audio element.
</audio>


Don't desert me here in the desert!

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00009.wav" autoplay/>
Your browser does not support the audio element.
</audio>


He thought it was time to present the present.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00010.wav" autoplay/>
Your browser does not support the audio element.
</audio>

Thisss isrealy awhsome.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00011.wav" autoplay/>
Your browser does not support the audio element.
</audio>


Punctuation sensitivity, is working.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00012.wav" autoplay/>
Your browser does not support the audio element.
</audio>


Punctuation sensitivity is working.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00013.wav" autoplay/>
Your browser does not support the audio element.
</audio>


The buses aren't the problem, they actually provide a solution.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00014.wav" autoplay/>
Your browser does not support the audio element.
</audio>


The buses aren't the PROBLEM, they actually provide a SOLUTION.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00015.wav" autoplay/>
Your browser does not support the audio element.
</audio>


The quick brown fox jumps over the lazy dog.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00016.wav" autoplay/>
Your browser does not support the audio element.
</audio>

Does the quick brown fox jump over the lazy dog?

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00017.wav" autoplay/>
Your browser does not support the audio element.
</audio>


Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00018.wav" autoplay/>
Your browser does not support the audio element.
</audio>


She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00019.wav" autoplay/>
Your browser does not support the audio element.
</audio>


The blue lagoon is a nineteen eighty American romance adventure film.

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/tacotron2/20180510_mixture_lj_checkpoint_step000320000_ema_speech-mel-00020.wav" autoplay/>
Your browser does not support the audio element.
</audio>


### On-line demo

A demonstration notebook supposed to be run on Google colab can be found at [Tacotron2: WaveNet-basd text-to-speech demo](https://colab.research.google.com/github/r9y9/Colaboratory/blob/master/Tacotron2_and_WaveNet_text_to_speech_demo.ipynb).


## References

- [Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", 	arXiv:1609.03499, Sep 2016.](https://arxiv.org/abs/1609.03499)
- [Aaron van den Oord, Yazhe Li, Igor Babuschkin, et al, "Parallel WaveNet: Fast High-Fidelity Speech Synthesis", 	arXiv:1711.10433, Nov 2017.](https://arxiv.org/abs/1711.10433)
- [Tamamori, Akira, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0314.PDF)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
- [Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.](https://arxiv.org/abs/1710.07654)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", 	arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
