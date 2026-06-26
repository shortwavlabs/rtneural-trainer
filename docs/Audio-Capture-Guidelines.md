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

The prepare step estimates target latency with transient-aware active-window
correlation. It scores rectified amplitude, pre-emphasized signal detail, onset
shape, and a small signed-correlation term, then reports how many analysis
windows agreed on each candidate. Check both confidence and window agreement
before long training runs.

Guidelines:

- Use the automatic latency estimate as a starting point.
- For moderate or low confidence, sweep manual latency around the estimate and
  compare residuals.
- When the app shows candidate offsets in Align, try those exact candidates
  before committing to a long WaveNet run.
- Treat low window agreement as a real ambiguity signal, even when the top
  numeric score looks plausible.
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

Start the DI with 2-3 seconds of clear, dry, varied attacks before the main
musical pass. Palm mutes, single-note plucks, and hard/soft pick attacks are
ideal. This gives the latency estimator crisp transient evidence before the
capture settles into dense rhythm, lead, or sustained material.

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

Lead captures need extra discipline even when they are pure amp-head captures.
The current lead target had no cabinet, delay, reverb, modulation, gate,
limiter, or post-amp compressor, yet it still had the lowest crest factor and
weakest latency confidence in the set. Treat dense lead saturation as its own
hard case: use the transient pre-roll, review latency candidates, and compare
balanced versus quality before assuming the capture is bad.

## Preset Expectations From Current Captures

The current clean/crunch/rhythm/edge/lead/overdrive-pedal experiments all used
the same 151.6 second DI2 performance. The practical training rule is now:

- Start amp and pedal captures with `wavenet_tcn_balanced`.
- Use `wavenet_tcn_quality` when maximum fidelity matters, when balanced leaves
  audible residual detail, or for dense crunch/rhythm/pedal tones.
- Try `wavenet_tcn_quality_tanh15` when a quality WaveNet sounds close but the
  residual or export warning points to high-band fizz/aliasing and the A2 PReLU
  runtime cost is too high.
- Try `wavenet_tcn_a2_prelu` for dense high-gain rhythm when quality/tanh15
  leaves audible upper-band residual or ASR warnings. On RHYTHM4, one A2 PReLU
  run beat the best multi-run tanh15 result and cut average ASR roughly in half.
  It borrows A2-style dilations, mixed kernels, and PReLU, but it is not a
  direct NAM A2 graph.
- Treat `wavenet_tcn_high_gain` as hidden research only. The first DI4/RHYTHM4
  test showed the longer sequential tanh stack underpowered the prediction and
  did not beat `wavenet_tcn_quality`.
- Use `wavenet_tcn_fast` as a quick WaveNet probe, not as the final model for
  hard captures.

The product UI is now WaveNet-only. Dense, recurrent, and smaller Conv1D
architectures may still exist as internal RTNeural fixture coverage, but they
are no longer recommended as capture-quality paths.

Balanced and quality were both excellent on clean, edge-of-breakup, and
overdrive pedal captures. Quality clearly won crunch and rhythm. Lead was the
outlier: balanced and quality tied, and balanced was the practical winner
because it had better runtime margin.

Second-generation DI3 findings refine this rule:

- A long, varied DI can work very well. `DI3.wav` plus the re-exported
  `CLEAN3.wav` trained cleanly with WaveNet balanced and exported with excellent
  native parity, comfortable RTNeural runtime, and low ASR.
- Very long captures train more slowly because the trainer samples windows
  across the file. For production captures, 90-180 seconds remains the better
  default unless the extra material is clearly useful.
- DI4/RHYTHM4 confirmed the practical upper bound: about 2.5-4 minutes is
  enough for a serious high-gain run when the performance includes varied
  picking, palm mutes, single notes, chords, harmonics, and transitions.
- For dense raw amp-head rhythm tones, WaveNet balanced can still underfit even
  after trimming silence. The `DI3-B_1.wav` / `RHYTHM3-B.wav` pair only became
  a good candidate with `wavenet_tcn_quality`.
- The transient preamble is still worth doing, but it does not make latency
  trivial for heavy tones. `RHYTHM3-B.wav` estimated `10 samples` with only
  `0.40` confidence and `42%` window agreement, so heavy captures still deserve
  manual review before long training.
- The app now preserves prepared audio as 32-bit float WAV. That avoids
  avoidable quantization in prep and is appropriate for captures exported from
  DAWs such as Logic.

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
The top latency candidates are close or too few analysis windows agreed on one
offset. Try the candidate offsets shown in Align before long runs, especially
for rhythm and lead tones.

`long_capture`:
The capture is long enough that the trainer will sample windows across it. Use a
larger window budget for better coverage.

`review_aliasing`:
The RTNeural export passed parity and benchmark checks, but sine-probe ASR was
elevated. This is an export/model diagnostic, not a capture-format failure.
Listen for metallic foldback or unnatural fizz on sustained high notes,
harmonics, and high-register bends. The current rhythm quality export measured
worst ASR `0.0678` at the ~`5 kHz` probe, which is in the review band
(`0.02-0.08`), not the high-aliasing band. It should be accepted or rejected by
listening, not by the ASR warning alone.

ASR warnings are more likely on raw high-gain amp-head captures because there is
no cabinet rolloff to hide upper harmonics. A future plugin should test
oversampling or higher-rate models for these exports, but the capture habit is
still the same: avoid clipping, keep levels consistent, and record enough clean
transient evidence for alignment.

## Current DI3 Calibration Notes

The DI3 family is more varied and longer than the DI2 family. It is useful for
stress-testing production behavior, but it also shows where we should avoid
over-recording.

| Pair | Duration | DI peak / RMS | Target peak / RMS | Latency | Training read |
| --- | ---: | ---: | ---: | ---: | --- |
| `DI3.wav` / `CLEAN3.wav` | `613.08 s` | `-5.56 / -31.50 dBFS` | `-5.09 / -23.63 dBFS` | known `13 samples` | WaveNet balanced exported as an excellent clean candidate. |
| `DI3-B_1.wav` / `RHYTHM3-B.wav` | `444.25 s` | `-5.44 / -29.86 dBFS` | `-9.49 / -20.25 dBFS` | estimated `10 samples`, low confidence | WaveNet balanced underfit; WaveNet quality reached a good export candidate. |

Takeaways:

- The DI level is healthy and unclipped.
- The heavy rhythm target is quieter by peak but much louder by RMS, which is
  expected for dense saturation.
- Long captures need higher window budgets, but length alone does not solve a
  hard high-gain tone. Model capacity and latency review still matter.
- A shorter focused DI4-style capture is worth testing against the same target
  class to see whether it reaches similar quality with less training time.

## Current DI2 Calibration Notes

The latest calibration family used one DI with six outputs:

| Target | Target peak | Target RMS | RMS delta vs DI | Estimated latency | Confidence | Training read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CLEAN2 | n/a | `-22.48 dBFS` | `+1.58 dB` | `12 samples` | `0.92` | WaveNet balanced/quality both excellent. |
| CRUNCH2 | n/a | `-17.27 dBFS` | `+6.79 dB` | `6 samples` | `0.76` | WaveNet quality clearly preferred. |
| RHYTHM2 | n/a | `-16.40 dBFS` | `+7.66 dB` | `10 samples` | `0.60` | Hardest amp capture; try latency candidates before long runs. |
| EDGE2 | `-5.59 dBFS` | `-18.26 dBFS` | `+5.80 dB` | `11 samples` | `0.88` | Healthy edge-of-breakup case; WaveNet quality best, balanced excellent. |
| LEAD2 | `-9.74 dBFS` | `-14.95 dBFS` | `+9.11 dB` | `9 samples` | `0.59` | Dense pure-amp lead outlier; balanced tied quality, review latency. |
| DRIVE2 | `-5.00 dBFS` | `-15.09 dBFS` | `+8.97 dB` | `3 samples` | `0.85` | Strong overdrive pedal capture; quality best, balanced excellent. |

Takeaways:

- A dry DI around `-5.3 dBFS` peak and `-24.1 dBFS` RMS worked well.
- Processed target RMS from `-22.5` to `-15 dBFS` can be valid when it reflects
  the real rig behavior.
- Low confidence around `0.60` is not a failure, but it should trigger latency
  candidate checks.
- Pure amp-head lead tones can still be harder than rhythm or pedal captures
  when crest factor is low and latency candidates are close.
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
