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

1. WN conditioned on mel-spectrogram (16-bit linear PCM, 22.5kHz)
2. WN conditioned on mel-spectrogram (8-bit mu-law, 16kHz)
3. WN conditioned on mel-spectrogram and speaker-embedding (16-bit linear PCM, 16kHz)
3. (Not yet) DeepVoice3 + WaveNet vocoder

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
| Input type | 8-bit mu-law encoded one-hot vector |
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

## DeepVoice3 + WaveNet vocoder

TODO

## References

- [Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", 	arXiv:1609.03499, Sep 2016.](https://arxiv.org/abs/1609.03499)
- [Aaron van den Oord, Yazhe Li, Igor Babuschkin, et al, "Parallel WaveNet: Fast High-Fidelity Speech Synthesis", 	arXiv:1711.10433, Nov 2017.](https://arxiv.org/abs/1711.10433)
- [Tamamori, Akira, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0314.PDF)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
- [Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.](https://arxiv.org/abs/1710.07654)
