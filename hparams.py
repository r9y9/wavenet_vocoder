import tensorflow as tf

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="wavenet_vocoder",

    # Convenient model builder
    builder="wavenet",

    # Presets known to work good.
    # NOTE: If specified, override hyper parameters with preset
    preset="",
    presets={
    },

    # If mulaw is True, audio signal in [-1, 1] is mu-law quantized to
    # [0, quantize_channels), conveted to one-hot vector and
    # then fed to the network, otherwise fed directly to the network.
    # NOTE: if you change the one of the two parameters below, then you need to
    # re-run preprocessing.
    mulaw=False,
    quantize_channels=256,  # 65536 or 256

    # Audio:
    sample_rate=16000,
    # this is only valid for mulaw is True
    silence_threshold=2,
    num_mels=80,
    fft_size=1024,
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,
    frame_shift_ms=None,
    min_level_db=-100,
    ref_level_db=20,

    # Model:
    # This should equal to `quantize_channels` if mulaw enabled
    # otherwise num_mixture * 3 (pi, mean, log_scale)
    out_channels=10 * 3,
    layers=20,
    stacks=2,
    residual_channels=512,
    gate_channels=512,  # split into 2 gropus internally for gated activation
    skip_out_channels=256,
    dropout=1 - 0.95,
    kernel_size=3,
    # If True, apply weight normalization as same as DeepVoice3
    weight_normalization=True,

    # Local conditioning (set negative value to disable))
    cin_channels=80,
    # If True, use transposed convolutions to upsample conditional features,
    # otherwise repeat features to adjust time resolution
    upsample_conditional_features=True,
    # should np.prod(upsample_scales) == hop_size
    upsample_scales=[16, 16],
    # Freq axis kernel size for upsampling network
    freq_axis_kernel_size=3,

    # Global conditioning (set negative value to disable)
    # currently limited for speaker embedding
    # this should only be enabled for multi-speaker dataset
    gin_channels=-1,  # i.e., speaker embedding dim
    n_speakers=7,  # 7 for CMU ARCTIC

    # Data loader
    pin_memory=True,
    num_workers=2,

    # train/test
    # test size can be specified as portion or num samples
    test_size=0.0441,  # 50 for CMU ARCTIC single speaker
    test_num_samples=None,
    random_state=1234,

    # Loss

    # Training:
    batch_size=2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    initial_learning_rate=1e-3,
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=1.0,
    # max time steps can either be specified as sec or steps
    # This is needed for those who don't have huge GPU memory...
    # if both are None, then full audio samples are used
    max_time_sec=None,
    max_time_steps=8000,

    # Save
    # per-step intervals
    checkpoint_interval=10000,
    train_eval_interval=10000,
    # per-epoch interval
    test_eval_epoch_interval=5,
    save_optimizer_state=True,

    # Eval:
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
