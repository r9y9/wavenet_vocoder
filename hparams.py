from wavenet_vocoder.tfcompat.hparam import HParams
import numpy as np

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
    name="wavenet_vocoder",

    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot
    # input and softmax output are assumed.
    # **NOTE**: if you change the one of the two parameters below, you need to
    # re-run preprocessing before training.
    input_type="raw",
    quantize_channels=65536,  # 65536 or 256

    # Audio:
    # time-domain pre/post-processing
    # e.g., preemphasis/inv_preemphasis
    # ref: LPCNet https://arxiv.org/abs/1810.11846
    preprocess="",
    postprocess="",
    # waveform domain scaling
    global_gain_scale=1.0,

    sample_rate=22050,
    # this is only valid for mulaw is True
    silence_threshold=2,
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=1024,
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,
    frame_shift_ms=None,
    win_length=1024,
    win_length_ms=-1.0,
    window="hann",

    # DC removal
    highpass_cutoff=70.0,

    # Parametric output distribution type for scalar input
    # 1) Logistic or 2) Normal
    output_distribution="Logistic",
    log_scale_min=-16.0,

    # Model:
    # This should equal to `quantize_channels` if mu-law quantize enabled
    # otherwise num_mixture * 3 (pi, mean, log_scale)
    # single mixture case: 2
    out_channels=10 * 3,
    layers=24,
    stacks=4,
    residual_channels=128,
    gate_channels=256,  # split into 2 gropus internally for gated activation
    skip_out_channels=128,
    dropout=0.0,
    kernel_size=3,

    # Local conditioning (set negative value to disable))
    cin_channels=80,
    cin_pad=2,
    # If True, use transposed convolutions to upsample conditional features,
    # otherwise repeat features to adjust time resolution
    upsample_conditional_features=True,
    upsample_net="ConvInUpsampleNetwork",
    upsample_params={
        "upsample_scales": [4, 4, 4, 4],  # should np.prod(upsample_scales) == hop_size
    },

    # Global conditioning (set negative value to disable)
    # currently limited for speaker embedding
    # this should only be enabled for multi-speaker dataset
    gin_channels=-1,  # i.e., speaker embedding dim
    n_speakers=7,  # 7 for CMU ARCTIC

    # Data loader
    pin_memory=True,
    num_workers=2,

    # Loss

    # Training:
    batch_size=8,
    optimizer="Adam",
    optimizer_params={
        "lr": 1e-3,
        "eps": 1e-8,
        "weight_decay": 0.0,
    },

    # see lrschedule.py for available lr_schedule
    lr_schedule="step_learning_rate_decay",
    lr_schedule_kwargs={"anneal_rate": 0.5, "anneal_interval": 200000},

    max_train_steps=1000000,
    nepochs=2000,

    clip_thresh=-1,

    # max time steps can either be specified as sec or steps
    # if both are None, then full audio samples are used in a batch
    max_time_sec=None,
    max_time_steps=10240,  # 256 * 40

    # Hold moving averaged parameters and use them for evaluation
    exponential_moving_average=True,
    # averaged = decay * averaged + (1 - decay) * x
    ema_decay=0.9999,

    # Save
    # per-step intervals
    checkpoint_interval=100000,
    train_eval_interval=100000,
    # per-epoch interval
    test_eval_epoch_interval=50,
    save_optimizer_state=True,

    # Eval:
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
