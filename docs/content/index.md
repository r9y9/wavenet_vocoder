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

This page provides audio samples for the open source implementation of the WaveNet vocoder.

1. WaveNet vocoder conditioned on mel-spectrogram
2. (Not yet) WaveNet vocoder conditioned on mel-spectrogram and speaker embedding
3. (Not yet) DeepVoice3 + WaveNet vocoder

## WaveNet vocoder conditioned on mel-spectrogram

| key                         | value |
|---------------------------------|------------------------------------------------------|
| Data                            | CMU ARCTIC (`clb`) (1183 for training, 50 for testing) |
| Sampling frequency  | 16kHz |
| Local conditioning            | 80-dim mel-spectrogram                               |
| Global conditioning            | N/A                              |
| Total layers                    | 16                                                   |
| Num cycles                      | 2                                                    |
| Receptive field (samples / ms) | 1021 / 63.8                                          |

<br/>

Samples from a model trained for 100k steps (~22 hours)

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/0_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/1_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/2_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/3_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/4_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/5_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/6_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/7_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/8_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<audio controls="controls" >
<source src="/wavenet_vocoder/audio/slt/9_checkpoint_step000100000_predicted.wav" autoplay/>
Your browser does not support the audio element.
</audio>

## WaveNet vocoder conditioned on mel-spectrogram and speaker embedding

| key                         | value |
|---------------------------------|------------------------------------------------------|
| Data                            | CMU ARCTIC (7580 for training, 350 for testing) |
| Local conditioning            | 80-dim mel-spectrogram                               |
| Global conditioning            | 4-dim speaker embedding                              |
| Total layers                    | 16                                                   |
| Num cycles                      | 2                                                    |
| Receptive field (samples / ms) | 1021 / 63.8                                          |

TODO

## DeepVoice3 + WaveNet vocoder

TODO

## References

- [Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", 	arXiv:1609.03499, Sep 2016.](https://arxiv.org/abs/1609.03499)
- [Aaron van den Oord, Yazhe Li, Igor Babuschkin, et al, "Parallel WaveNet: Fast High-Fidelity Speech Synthesis", 	arXiv:1711.10433, Nov 2017.](https://arxiv.org/abs/1711.10433)
- [Tamamori, Akira, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0314.PDF)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
- [Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.](https://arxiv.org/abs/1710.07654)
