# Clean, Crunch, Rhythm, Edge, Lead, And Pedal Capture Baseline

Reviewed: 2026-06-22

This note summarizes the first comparable clean, crunch, heavy rhythm,
edge-of-breakup, lead, and overdrive pedal training passes after the latest
WaveNet, learning-rate, preview, and report updates. All captures use the same
dry source file, `DI2.wav`, with 151.6 seconds of 48 kHz audio. Source files
were stereo dual-mono and prepared with the current mix-to-mono policy.

## Short Answer

WaveNet should be the primary quality lane across amp and overdrive pedal
captures. The expanded evidence does not support treating
`conv1d_stack_prelu` as the default medium/low-gain preset. It remains valuable
as a very fast CPU fallback and sanity check, but it underfit the amp captures
and trailed WaveNet on the overdrive pedal capture.

The practical default should be:

- Use `wavenet_tcn_balanced` as the first quality run for amp and pedal
  captures.
- Use `wavenet_tcn_quality` for crunch, rhythm, dense overdrive pedal captures,
  and any capture where balanced still leaves audible residual detail.
- Treat lead captures as a special review path when latency confidence is low
  or the target is very compressed.
- Keep `wavenet_tcn_fast` as a quick WaveNet probe.
- Keep `conv1d_stack_prelu` as a speed fallback, not the main amp-quality path.

## Capture Health

| Capture | Project | Target | Target RMS | RMS Delta vs DI | Latency | Confidence | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Clean | `project_0787cbe1cc784a10ad96a942d271ec4c` | `CLEAN2.wav` | `-22.48 dBFS` | `+1.58 dB` | `12 samples` | `0.92` | Healthy alignment and gain. |
| Crunch | `project_2bedf2386d4a4fd1981b346f4202701c` | `CRUNCH2.wav` | `-17.27 dBFS` | `+6.79 dB` | `6 samples` | `0.76` | Usable; more compressed/hotter than clean. |
| Rhythm | `project_98f406e8108d423ab624bc8ca5b1fcb7` | `RHYTHM2.wav` | `-16.40 dBFS` | `+7.66 dB` | `10 samples` | `0.60` | Hardest capture; latency candidates were close. |
| Edge | `project_5008ffab43db41469d9199ad8fa8292b` | `EDGE2.wav` | `-18.26 dBFS` | `+5.80 dB` | `11 samples` | `0.88` | Healthy edge-of-breakup case. |
| Lead | `project_c5c8cd0a208e42738a1962089c7349b4` | `LEAD2.wav` | `-14.95 dBFS` | `+9.11 dB` | `9 samples` | `0.59` | Dense/low-crest-factor lead tone; latency candidates were close. |
| Overdrive pedal | `project_87cf7d63df6c4d9dae65d41875211560` | `DRIVE2.wav` | `-15.09 dBFS` | `+8.97 dB` | `3 samples` | `0.85` | Strong pedal capture; clean latency estimate. |

No capture clipped. Rhythm produced a latency-review warning because the top
candidates were close: `10`, `2`, and `18` samples. Lead also produced a
latency-review warning: `9`, `1`, and `17` samples were very close. Post-training
preview shift searches over `+/-64` samples found best shift `0` for rhythm and
lead runs, so the final preview errors do not look like simple evaluation
offsets. Still, for future heavy-gain or lead finalization it is worth trying
manual latency candidates when prep confidence is this low.

## Results

The ESR values below are the app's preview ESR metrics. Lower is better.
Residual RMS is measured from the generated preview residual WAV.

### Clean

| Preset | Preview ESR | RMSE | Corr | Residual RMS | Est. RTF | Best Epoch | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_quality` | `0.0052` | `0.0082` | `0.9974` | `-41.69 dBFS` | `1.5x` | `118` | Best metric result. |
| `wavenet_tcn_balanced` | `0.0058` | `0.0087` | `0.9971` | `-41.18 dBFS` | `3.0x` | `117` | Nearly tied with quality; better runtime margin. |
| `wavenet_tcn_fast` | `0.0382` | `0.0223` | `0.9809` | `-33.03 dBFS` | `8.0x` | `8` | Good quick probe, early-stopped. |
| `conv1d_stack_prelu` | `0.0539` | `0.0265` | `0.9739` | `-31.53 dBFS` | `120x` | `43` | Very fast but audibly less faithful candidate. |

Clean is the only capture where balanced and quality are close enough that
runtime may decide the recommendation. `wavenet_tcn_balanced` is likely the best
default clean-tone quality preset unless native export benchmarking shows the
quality model has enough margin and the user wants maximum fidelity.

### Crunch

| Preset | Preview ESR | RMSE | Corr | Residual RMS | Est. RTF | Best Epoch | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_quality` | `0.0115` | `0.0190` | `0.9943` | `-34.44 dBFS` | `1.5x` | `119` | Clear best result. |
| `wavenet_tcn_balanced` | `0.0228` | `0.0267` | `0.9887` | `-31.46 dBFS` | `3.0x` | `120` | Good practical runner-up. |
| `wavenet_tcn_fast` | `0.0376` | `0.0343` | `0.9815` | `-29.29 dBFS` | `8.0x` | `120` | Useful speed probe. |
| `conv1d_stack_prelu` | `0.1015` | `0.0564` | `0.9495` | `-24.98 dBFS` | `120x` | `120` | Underfits crunch behavior. |

Crunch strengthens the WaveNet recommendation. The quality model has a high
`peak_residual` (`0.625`) but much better residual RMS and correlation than the
alternatives. Report language should distinguish isolated transient peaks from
sustained residual energy so a strong crunch model is not described too harshly.

### Rhythm

| Preset | Preview ESR | RMSE | Corr | Residual RMS | Est. RTF | Best Epoch | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_quality` | `0.1100` | `0.0529` | `0.9438` | `-25.53 dBFS` | `1.5x` | `119` | Best available result; worth continuation. |
| `wavenet_tcn_balanced` | `0.1727` | `0.0663` | `0.9105` | `-23.57 dBFS` | `3.0x` | `120` | Better than fast/stacked, but not final quality. |
| `wavenet_tcn_fast` | `0.2892` | `0.0858` | `0.8467` | `-21.33 dBFS` | `8.0x` | `120` | Too small for this capture. |
| `conv1d_stack_prelu` | `0.3952` | `0.1003` | `0.7788` | `-19.98 dBFS` | `120x` | `85` | Underfit and early-stopped. |

Rhythm is much harder than clean or crunch. The quality model is still the clear
winner, but the error floor is higher and the stream-validation metrics are
substantially worse than the preview metrics. This suggests a mix of higher
model-capacity demand, dense nonlinear behavior, and possibly capture segments
whose dynamics are not evenly represented by the current training windows.

### Edge Of Breakup

| Preset | Preview ESR | RMSE | Corr | Residual RMS | Est. RTF | Best Epoch | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_quality` | `0.0045` | `0.0118` | `0.9978` | `-38.54 dBFS` | `1.5x` | `105` | Best metric result. |
| `wavenet_tcn_balanced` | `0.0092` | `0.0170` | `0.9954` | `-35.39 dBFS` | `3.0x` | `114` | Excellent practical runner-up. |
| `wavenet_tcn_fast` | `0.0582` | `0.0427` | `0.9709` | `-27.40 dBFS` | `8.0x` | `120` | Good quick probe. |
| `conv1d_stack_prelu` | `0.0771` | `0.0491` | `0.9608` | `-26.18 dBFS` | `120x` | `117` | Good but clearly behind WaveNet. |

Edge-of-breakup confirms the default change. Balanced is already excellent and
has strong runtime margin. Quality is the preferred model if the native
benchmark passes and the listening difference matters.

### Lead

| Preset | Preview ESR | RMSE | Corr | Residual RMS | Est. RTF | Best Epoch | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_balanced` | `0.0810` | `0.0553` | `0.9587` | `-25.14 dBFS` | `3.0x` | `120` | Best available result; good candidate. |
| `wavenet_tcn_quality` | `0.0821` | `0.0557` | `0.9582` | `-25.08 dBFS` | `1.5x` | `73` | Tied with balanced, but lower runtime margin. |
| `conv1d_stack_prelu` | `0.1798` | `0.0824` | `0.9064` | `-21.68 dBFS` | `120x` | `120` | Needs work. |
| `wavenet_tcn_fast` | `0.1969` | `0.0863` | `0.8989` | `-21.28 dBFS` | `8.0x` | `120` | Needs work. |

Lead is the first capture where quality does not beat balanced. The target is
very dense: `-14.95 dBFS` RMS with only about `5.2 dB` crest factor. Latency
confidence is low, but a preview shift search still selected shift `0`, so the
rendered residual is real model mismatch rather than a report alignment artifact.
The residual is broad-band and concentrated in the presence/fizz region
(`1-9 kHz`). This suggests dense lead saturation and possibly time-varying chain
behavior, not a bad WAV file.

### Overdrive Pedal

| Preset | Preview ESR | RMSE | Corr | Residual RMS | Est. RTF | Best Epoch | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_quality` | `0.0021` | `0.0095` | `0.9990` | `-40.49 dBFS` | `1.5x` | `118` | Best metric result; excellent. |
| `wavenet_tcn_balanced` | `0.0044` | `0.0136` | `0.9978` | `-37.32 dBFS` | `3.0x` | `120` | Excellent practical runner-up. |
| `wavenet_tcn_fast` | `0.0153` | `0.0254` | `0.9924` | `-31.90 dBFS` | `8.0x` | `119` | Good quick probe. |
| `conv1d_stack_prelu` | `0.0267` | `0.0335` | `0.9872` | `-29.49 dBFS` | `120x` | `120` | Good fallback, but not best. |

The overdrive pedal capture went very well. Latency confidence is high enough to
trust, and both balanced and quality are excellent. Quality wins the metrics,
while balanced may be the better product export if native benchmark margin or
CPU budget matters.

## Interpretation

1. WaveNet is not only a high-gain rescue preset.

   It won clean, crunch, rhythm, edge, lead, and overdrive pedal. On clean,
   edge, and overdrive pedal, balanced and quality were both excellent. On
   crunch and rhythm, quality opened a larger gap. Lead is the exception where
   balanced slightly beat quality.

2. Stacked Conv is currently a speed fallback.

   `conv1d_stack_prelu` is extremely fast, and it can produce good results on
   simpler captures, but it missed too much behavior on every amp capture in
   this set and trailed WaveNet on the overdrive pedal capture. It should not be
   the default quality recommendation unless future capture families prove a
   repeatable exception.

3. Rhythm needs more than one 120-epoch quality pass.

   `wavenet_tcn_quality` remained the best rhythm model and did not early-stop.
   Continue-from-best with a lower learning rate is justified, especially before
   comparing exports.

4. Lead needs a special review path.

   Lead had the hottest RMS, lowest crest factor, and weakest latency confidence
   in this set. Balanced and quality tied, and the residual stayed broad-band in
   the presence/fizz region. Future lead captures should try top latency
   candidates before long runs and, when possible, avoid post-amp delay, reverb,
   modulation, gates, or compressors in the training target.

5. Latency confidence should affect workflow language.

   Clean, edge, and pedal had high latency confidence. Crunch was usable. Rhythm
   and lead were low enough that the app correctly asked for review. The preview
   shift search did not show a simple offset, but future rhythm/lead/high-gain
   captures should encourage checking top latency candidates before spending long
   training time.

6. Report quality needs a fourth nuance: excellent/preferred.

   The current good/usable/needs-work buckets are useful, but they do not express
   ranking well. Clean, edge, and pedal WaveNet quality and balanced can all be
   export candidates, while crunch quality is clearly preferred over balanced.
   Lead is "good" but not "excellent", which matches its higher residual floor.
   Add language that separates "export candidate" from "best among this
   comparison".

## Product Changes Suggested By This Baseline

- Recommend `wavenet_tcn_balanced` as the default first quality run for amp
  captures.
- Recommend `wavenet_tcn_quality` automatically for crunch/high-gain captures,
  for any run where balanced leaves high residual RMS, or when the user chooses
  maximum fidelity.
- Keep `wavenet_tcn_balanced` as the practical winner when balanced and quality
  tie, especially for lead-like captures or runtime-sensitive exports.
- Demote `conv1d_stack_prelu` from "medium/low-gain default" to "fast CPU
  fallback / sanity check" until further evidence.
- Add report language that weighs residual RMS and correlation more strongly
  than isolated peak residual.
- Add latency-review workflow copy for low-confidence high-gain captures:
  "try top candidate offsets before long training".
- Export and native-benchmark both `wavenet_tcn_balanced` and
  `wavenet_tcn_quality` on captures where runtime margin matters.

## Follow-Up Implemented

- The default built-in training recipe now uses `wavenet_tcn_balanced`.
- `wavenet_tcn_quality` is the high-gain/refinement recipe, while
  `conv1d_stack_prelu` is labeled as a fast fallback.
- The Align view now extracts and surfaces top latency candidates from the
  prepare warning so low-confidence captures can be checked before long runs.
- Training reports now include an `excellent` tier and weigh residual RMS plus
  correlation more heavily than isolated peak residuals.
- Export packages now include parity snapshot WAVs and `parity-snapshot.json`
  alongside validation and benchmark reports.

## Next Tests

1. Continue rhythm `wavenet_tcn_quality` from best checkpoint with a lower
   learning rate and more epochs.
2. For lead, try the top latency candidates from preparation (`1`, `9`, and
   `17` samples) with shorter balanced runs before another long production run.
3. Export clean, edge, pedal, crunch, and rhythm balanced/quality winners;
   compare native benchmark matrix results.
4. If lead remains harder, record or test an amp/cab-only lead target without
   delay, reverb, modulation, gate, or compressor after the amp.
5. If rhythm still has audible residual, try the top latency candidates from
   preparation (`2`, `10`, and `18` samples) before another long run.
6. Add a second clean or edge-of-breakup capture to make sure the clean result
   was not specific to this amp setting.
