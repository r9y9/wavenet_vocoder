python train.py --preset=./presets/liepa_mixture.json --data-root=/Tacotron-2/tacotron_output --checkpoint=./checkpoints/checkpoint_step001210000_ema.pth --hparams="tacotron_convert=True" > output.log 2>&1 & disown -a