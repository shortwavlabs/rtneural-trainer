# Audio Capture Guidelines

These guidelines describe how to record dry input and processed target WAVs for
RTNeural Trainer. They are intentionally practical: good captures make training
faster, validation clearer, and exported models less surprising in a plugin or
runtime.

## Capture Pair Contract

Each training project needs two files:

1. A dry input recording.
2. A processed target recording made from the same performance through the amp,
   pedal, plugin, or signal chain being modeled.

The two files should:

- Have the same sample rate.
- Have the same channel layout, ideally mono.
- Start from the same performance with no trimming differences.
- Preserve any real device latency until the prepare step estimates or adjusts
  it.
- Be long enough to include the behavior you want the model to learn.

For v1, 48 kHz mono WAV is the safest target format. The app supports PCM WAV
and 32-bit float WAV captures.

Stereo dual-mono files are acceptable, but true mono is preferred. The app can
mix stereo files to mono, and this was harmless in the current DI2 experiments
because both channels were identical. If the left and right channels differ,
mixdown can change the effective tone, phase, or gain, so record mono whenever
repeatability matters.

## Recommended Levels

Aim for repeatable headroom, not maximum loudness.

Recommended starting targets:

- Dry input peak: roughly -12 to -6 dBFS.
- Processed target peak: roughly -12 to -3 dBFS.
- Avoid peaks above -1 dBFS.
- Avoid clipped samples.
- Keep average level high enough that the active material is well above the
  noise floor.

The DI2 capture family that produced the best results used a dry input around
`-5.3 dBFS` peak and `-24.1 dBFS` RMS. Successful targets ranged from roughly
`-22.5 dBFS` RMS for clean to roughly `-15 dBFS` RMS for lead and overdrive
pedal. That range is usable when intentional; the important thing is avoiding
clipping and keeping the capture gain consistent.

The app warns when either file has less than 1 dB of peak headroom. That warning
does not mean the file is unusable, but it does mean clipped transients or
near-clips may dominate the loss and make the trained model chase recording
artifacts.

## Output Level Consistency

For a profile family, keep recording gain consistent across all processed
targets unless the loudness difference is intentional.

Good practice:

- Set the interface input gain once.
- Set the device or plugin output once.
- Record all targets without changing capture gain.
- Write down any intentional output-level changes.
- Use the same DI file for every target in the family.

Do not normalize every target independently by default. Independent
normalization can hide the real output level of the rig and make exported models
feel wrong when switched in a runtime.

Use normalization only when:

- The original recording level was clearly a capture mistake.
- You apply the decision consistently and document it.
- You understand that the exported model will learn the normalized level, not
  the original rig level.

Clean tones can have much lower RMS than distorted tones because they are more
dynamic. Treat that as a review signal, not an automatic failure. A clean target
that is 6-8 dB quieter by RMS may be correct, but it should be intentional.

Dense lead and pedal captures may be 9 dB or more louder than the dry DI by RMS.
That is not automatically bad, but it usually means the model needs the WaveNet
quality lane and careful listening to the residual.

## Latency And Alignment

The prepare step estimates target latency with cross-correlation. Check the
reported confidence before long training runs.

Guidelines:

- Use the automatic latency estimate as a starting point.
- For moderate or low confidence, sweep manual latency around the estimate and
  compare residuals.
- When the app shows candidate offsets in Align, try those exact candidates
  before committing to a long WaveNet run.
- Do not assume every profile in a family has the same latency.
- Heavier processing paths may report different latency than clean or crunch
  paths.
- If the model predicts the right tone but residual peaks stay high, alignment
  should be one of the first things to check.

Current calibration:

- High-confidence captures around `0.85` and above have been reliable enough to
  trust for long runs.
- Around `0.75` is usable, but listen carefully.
- Around `0.60` or below should trigger a candidate sweep before long training.

For the DI2 family, rhythm confidence was `0.60` with candidates `10`, `2`, and
`18` samples, and lead confidence was `0.59` with candidates `9`, `1`, and `17`
samples. In both cases, post-training preview shift checks still preferred
shift `0`, so the final residual was not a simple evaluation offset. Even so,
trying candidate offsets before long training is worthwhile because alignment
can affect how easily the model learns.

## Capture Length

Short captures are useful for quick smoke tests, but real profiles need enough
active material.

Recommended minimums:

- Quick pipeline smoke: 5-15 seconds.
- Useful first training pass: 30-60 seconds.
- Stronger real-world capture: 90-180 seconds with varied playing.

For long captures, the app samples training windows across the file. Increase
the window budget when you want more coverage of a long performance.

For 90-180 second captures, `2048` windows is a good first pass and `4096` to
`8192` windows is more appropriate for final WaveNet balanced/quality runs,
especially for high-gain rhythm or dense lead captures.

## Source Material

Use material that excites the behavior you want the model to learn.

Include:

- Single notes across the register.
- Chords and double-stops.
- Palm-muted transients for rhythm tones.
- Sustained notes for compression and decay.
- Dynamics from soft to hard picking.
- Silence or near-silence only when noise behavior matters.
- A few transitions between muted, open, single-note, and chordal playing so
  sampled training windows see more than one behavior.

Avoid:

- Long stretches of silence in the middle of the capture.
- Clipped input or output.
- Different performances for dry and target.
- Time-based effects for v1 models unless you are deliberately testing an
  unsupported behavior.
- Changing knobs, pickup selection, input gain, or output gain mid-capture
  unless that variation is part of the intended model.

Lead captures need extra discipline. If possible, capture an amp/cab-only lead
target first: no delay, reverb, modulation, post-amp compressor, gate, or
limiter. The current lead capture trained to a usable model, but it had the
lowest crest factor and weakest latency confidence in the set. Time-varying
post effects can make this harder because the model is trying to learn one
fixed nonlinear system, not a whole mix-ready lead chain.

## Preset Expectations From Current Captures

The current clean/crunch/rhythm/edge/lead/overdrive-pedal experiments all used
the same 151.6 second DI2 performance. The practical training rule is now:

- Start amp and pedal captures with `wavenet_tcn_balanced`.
- Use `wavenet_tcn_quality` when maximum fidelity matters, when balanced leaves
  audible residual detail, or for dense crunch/rhythm/pedal tones.
- Treat `conv1d_stack_prelu` as a fast CPU fallback and sanity check, not the
  main quality lane.
- Use `wavenet_tcn_fast` as a quick WaveNet probe, not as the final model for
  hard captures.

Balanced and quality were both excellent on clean, edge-of-breakup, and
overdrive pedal captures. Quality clearly won crunch and rhythm. Lead was the
outlier: balanced and quality tied, and balanced was the practical winner
because it had better runtime margin.

## Interpreting Warnings

`capture_headroom_low`:
The dry or target file is too close to full scale. Recapture with more headroom
when possible.

`capture_level_low`:
The file peaks very low. Recapture hotter if the signal is close to the noise
floor.

`rms_mismatch`:
The dry and processed average levels are far apart. This may be intentional for
some rigs, but it should be reviewed before training.

`latency_estimate_review`:
The top latency candidates are close. Try the candidate offsets shown in Align
before long runs, especially for rhythm and lead tones.

`long_capture`:
The capture is long enough that the trainer will sample windows across it. Use a
larger window budget for better coverage.

## Current DI2 Calibration Notes

The latest calibration family used one DI with six outputs:

| Target | Target peak | Target RMS | RMS delta vs DI | Estimated latency | Confidence | Training read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CLEAN2 | n/a | `-22.48 dBFS` | `+1.58 dB` | `12 samples` | `0.92` | WaveNet balanced/quality both excellent. |
| CRUNCH2 | n/a | `-17.27 dBFS` | `+6.79 dB` | `6 samples` | `0.76` | WaveNet quality clearly preferred. |
| RHYTHM2 | n/a | `-16.40 dBFS` | `+7.66 dB` | `10 samples` | `0.60` | Hardest amp capture; try latency candidates before long runs. |
| EDGE2 | `-5.59 dBFS` | `-18.26 dBFS` | `+5.80 dB` | `11 samples` | `0.88` | Healthy edge-of-breakup case; WaveNet quality best, balanced excellent. |
| LEAD2 | `-9.74 dBFS` | `-14.95 dBFS` | `+9.11 dB` | `9 samples` | `0.59` | Dense lead outlier; balanced tied quality, review latency/chain. |
| DRIVE2 | `-5.00 dBFS` | `-15.09 dBFS` | `+8.97 dB` | `3 samples` | `0.85` | Strong overdrive pedal capture; quality best, balanced excellent. |

Takeaways:

- A dry DI around `-5.3 dBFS` peak and `-24.1 dBFS` RMS worked well.
- Processed target RMS from `-22.5` to `-15 dBFS` can be valid when it reflects
  the real rig behavior.
- Low confidence around `0.60` is not a failure, but it should trigger latency
  candidate checks.
- Lead tones are the most likely to hide non-modelable chain behavior. Capture a
  dry amp/cab-only lead target first when possible.
- Pedal captures can train extremely well with the same rules as amp captures
  when latency confidence is good.

## Historical 2025 Profile Family Notes

The first real profile-family check used one DI with four outputs:

| Target | Target peak | Target RMS | RMS delta vs DI | Estimated latency |
| --- | ---: | ---: | ---: | ---: |
| CLEAN | -1.35 dBFS | -29.70 dBFS | -7.62 dB | 12 samples |
| CRUNCH | -2.61 dBFS | -21.97 dBFS | +0.11 dB | 17 samples |
| RHYTHM | -5.86 dBFS | -21.84 dBFS | +0.25 dB | 105 samples |
| LEAD | -4.49 dBFS | -19.84 dBFS | +2.24 dB | 111 samples |

Takeaways:

- The DI is unclipped but has only about 0.09 dB of headroom, so a future
  capture should leave more dry-input margin.
- CRUNCH and RHYTHM are closely matched by RMS.
- LEAD is hotter, plausibly intentional for a lead profile.
- CLEAN is much quieter by RMS and should be checked for intentionality before
  spending on long training runs.
- Latency appears profile-dependent, so manual latency review should happen per
  target.
