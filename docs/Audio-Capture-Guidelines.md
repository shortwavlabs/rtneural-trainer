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

## Recommended Levels

Aim for repeatable headroom, not maximum loudness.

Recommended starting targets:

- Dry input peak: roughly -12 to -6 dBFS.
- Processed target peak: roughly -12 to -3 dBFS.
- Avoid peaks above -1 dBFS.
- Avoid clipped samples.
- Keep average level high enough that the active material is well above the
  noise floor.

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

## Latency And Alignment

The prepare step estimates target latency with cross-correlation. Check the
reported confidence before long training runs.

Guidelines:

- Use the automatic latency estimate as a starting point.
- For moderate or low confidence, sweep manual latency around the estimate and
  compare residuals.
- Do not assume every profile in a family has the same latency.
- Heavier processing paths may report different latency than clean or crunch
  paths.
- If the model predicts the right tone but residual peaks stay high, alignment
  should be one of the first things to check.

## Capture Length

Short captures are useful for quick smoke tests, but real profiles need enough
active material.

Recommended minimums:

- Quick pipeline smoke: 5-15 seconds.
- Useful first training pass: 30-60 seconds.
- Stronger real-world capture: 90-180 seconds with varied playing.

For long captures, the app samples training windows across the file. Increase
the window budget when you want more coverage of a long performance.

## Source Material

Use material that excites the behavior you want the model to learn.

Include:

- Single notes across the register.
- Chords and double-stops.
- Palm-muted transients for rhythm tones.
- Sustained notes for compression and decay.
- Dynamics from soft to hard picking.
- Silence or near-silence only when noise behavior matters.

Avoid:

- Long stretches of silence in the middle of the capture.
- Clipped input or output.
- Different performances for dry and target.
- Time-based effects for v1 models unless you are deliberately testing an
  unsupported behavior.
- Changing knobs, pickup selection, input gain, or output gain mid-capture
  unless that variation is part of the intended model.

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

`long_capture`:
The capture is long enough that the trainer will sample windows across it. Use a
larger window budget for better coverage.

## Current 2025 Profile Family Notes

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
