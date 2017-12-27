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

    # Audio:
    sample_rate=8000,

    # Model:
    channels=64,
    dropout=1 - 0.95,
    kernel_size=3,

    # Data loader
    pin_memory=True,
    num_workers=2,

    # Loss

    # Training:
    batch_size=4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    initial_learning_rate=1e-4,
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=0.1,

    # Save
    checkpoint_interval=1000,
    eval_interval=1000,
    save_optimizer_state=True,

    # Eval:
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
