# PANAMA / WaveNet Amp Modeling Findings

Date reviewed: June 21, 2026

Implementation update: RTNeural Trainer now ships RTNeural-safe WaveNet/TCN
presets (`wavenet_tcn_fast`, `wavenet_tcn_balanced`,
`wavenet_tcn_quality`), uses WaveNet balanced as the default amp quality
recipe, and keeps older Conv/LSTM/GRU/Dense paths only as internal exporter
fixtures. Parametric/active-learning capture remains deferred.

Primary paper: [Parametric Neural Amp Modeling with Active Learning](https://arxiv.org/html/2507.02109v1)

Related references reviewed:

- [PANAMA implementation](https://github.com/ETH-DISCO/PANAMA)
- [PANAMA WaveNet config](https://raw.githubusercontent.com/ETH-DISCO/PANAMA/main/default_config_files/models/wavenet-mel-mrstft.json)
- [PANAMA LSTM config](https://raw.githubusercontent.com/ETH-DISCO/PANAMA/main/default_config_files/models/lstm-mel-mrstft.json)
- [Deep Learning for Tube Amplifier Emulation](https://arxiv.org/abs/1811.00334)
- [Efficient neural networks for real-time modeling of analog dynamic range compression](https://arxiv.org/abs/2102.06200)
- [End-to-End Amp Modeling: From Data to Controllable Guitar Amplifier Models](https://arxiv.org/abs/2403.08559)

## Summary

PANAMA is about parametric amp modeling: one model is conditioned on amp knob
settings, and an active-learning loop chooses the next knob settings to record.
That is broader than our current non-parametric workflow, where one project
captures one fixed amp/pedal setting. Still, the paper is useful because it
validates several decisions for this app:

1. Feed-forward WaveNet/TCN models are a first-class architecture for amp
   modeling, not just a fallback for recurrent models.
2. Dilated causal convolutions are a practical way to increase temporal
   receptive field while staying finite-memory and RTNeural-friendly.
3. Spectral loss is important. PANAMA's recommended configs use mel /
   multi-resolution STFT weighting, and the earlier tube-amp WaveNet paper used
   pre-emphasis to help the model learn high-frequency content.
4. Alignment still matters. PANAMA's LSTM config notes that a delay error of
   even a few samples can hurt results, matching what this app already exposes
   in the alignment view.

## Immediate Product Impact

The paper supports adding a WaveNet-style preset to the desktop app. For V1,
the preset should stay RTNeural JSON compatible, so it should be a sequential
stack of causal dilated `Conv1D` layers with `tanh` activations and a bounded
output layer. Full WaveNet residual/skip/gated blocks can be revisited only
after the exporter and native validator support that graph safely.

The latest real capture run also supports this direction. The
`conv1d_stack_prelu` preset improved the DI2/RHYTHM2 project substantially over
the earlier `conv1d_bn_prelu` run, but it still left upper-band residual energy.
That historical result pushed the project toward WaveNet: increase finite-memory
receptive field and train with spectral pressure, instead of exposing smaller
Conv or recurrent presets as product choices.

## What To Build Now

- Keep testing `wavenet_tcn_balanced` and `wavenet_tcn_quality` across more amp
  and pedal captures.
- Keep the model finite-memory for stable continuous inference.
- Keep multi-resolution STFT + pre-emphasis as the WaveNet default loss path.
- Use latency candidate review before long high-gain runs when preparation
  confidence is low.
- Keep non-WaveNet presets as internal RTNeural export/parity fixtures only.

## What To Defer

- Parametric knob-conditioned models.
- Active-learning capture guidance over gain/bass/mid/treble/presence/master.
- Full WaveNet residual, skip, and gated blocks.
- Mel-scale loss if it requires extra dependency or exporter complexity.

Those are promising V2 directions, but the current app needs one-setting
captures to become reliable before asking users to record a grid of amp knob
positions.

## Capture Guidance Implications

For amp/pedal captures, keep emphasizing:

- Accurate latency alignment.
- Healthy but unclipped target level.
- Long enough diverse playing material.
- Sufficient high-frequency excitation for distorted captures.
- Listening checks against target, prediction, and residual, not ESR alone.

The paper does not change the recommended file format: 32-bit float WAV remains
appropriate.
