# Clean, Crunch, And Rhythm Capture Baseline

Reviewed: 2026-06-22

This note summarizes the first comparable clean, crunch, and heavy rhythm
training pass after the latest WaveNet, learning-rate, preview, and report
updates. All three captures use the same dry source file, `DI2.wav`, with
151.6 seconds of 48 kHz audio. Source files were stereo and prepared with the
current mix-to-mono policy.

## Short Answer

WaveNet should be the primary quality lane across clean, crunch, and heavy
rhythm amp captures. The current evidence does not support treating
`conv1d_stack_prelu` as the default medium/low-gain amp preset. It remains
valuable as a very fast CPU fallback and sanity check, but it underfit all three
captures compared with WaveNet.

The practical default should be:

- Use `wavenet_tcn_balanced` as the first quality run for clean and possibly
  lower-gain captures.
- Use `wavenet_tcn_quality` for crunch, rhythm, and any capture where balanced
  still leaves audible residual detail.
- Keep `wavenet_tcn_fast` as a quick WaveNet probe.
- Keep `conv1d_stack_prelu` as a speed fallback, not the main amp-quality path.

## Capture Health

| Capture | Project | Target | Target RMS | RMS Delta vs DI | Latency | Confidence | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Clean | `project_0787cbe1cc784a10ad96a942d271ec4c` | `CLEAN2.wav` | `-22.48 dBFS` | `+1.58 dB` | `12 samples` | `0.92` | Healthy alignment and gain. |
| Crunch | `project_2bedf2386d4a4fd1981b346f4202701c` | `CRUNCH2.wav` | `-17.27 dBFS` | `+6.79 dB` | `6 samples` | `0.76` | Usable; more compressed/hotter than clean. |
| Rhythm | `project_98f406e8108d423ab624bc8ca5b1fcb7` | `RHYTHM2.wav` | `-16.40 dBFS` | `+7.66 dB` | `10 samples` | `0.60` | Hardest capture; latency candidates were close. |

No capture clipped. Rhythm produced a latency-review warning because the top
candidates were close: `10`, `2`, and `18` samples. A post-training preview
shift search over `+/-64` samples found best shift `0` for every rhythm run, so
the final preview errors do not look like a simple evaluation offset. Still, for
future heavy-gain finalization it is worth trying manual latency candidates when
prep confidence is this low.

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

## Interpretation

1. WaveNet is not only a high-gain rescue preset.

   It won clean, crunch, and rhythm. On clean, balanced and quality were nearly
   tied. On crunch and rhythm, quality opened a larger gap.

2. Stacked Conv is currently a speed fallback.

   `conv1d_stack_prelu` is extremely fast, but it missed too much behavior on
   every amp capture in this set. It should not be the default quality
   recommendation for real amp captures until another capture family proves
   otherwise.

3. Rhythm needs more than one 120-epoch quality pass.

   `wavenet_tcn_quality` remained the best rhythm model and did not early-stop.
   Continue-from-best with a lower learning rate is justified, especially before
   comparing exports.

4. Latency confidence should affect workflow language.

   Clean had high latency confidence. Crunch was usable. Rhythm was low enough
   that the app correctly asked for review. The preview shift search did not
   show a simple offset, but future rhythm/high-gain captures should encourage
   checking top latency candidates before spending long training time.

5. Report quality needs a fourth nuance: excellent/preferred.

   The current good/usable/needs-work buckets are useful, but they do not express
   ranking well. Clean WaveNet quality and balanced are both "good", while
   crunch quality is clearly preferred over balanced. Add language that separates
   "export candidate" from "best among this comparison".

## Product Changes Suggested By This Baseline

- Recommend `wavenet_tcn_balanced` as the default first quality run for amp
  captures.
- Recommend `wavenet_tcn_quality` automatically for crunch/high-gain captures,
  for any run where balanced leaves high residual RMS, or when the user chooses
  maximum fidelity.
- Demote `conv1d_stack_prelu` from "medium/low-gain default" to "fast CPU
  fallback / sanity check" until further evidence.
- Add report language that weighs residual RMS and correlation more strongly
  than isolated peak residual.
- Add latency-review workflow copy for low-confidence high-gain captures:
  "try top candidate offsets before long training".
- Export and native-benchmark both `wavenet_tcn_balanced` and
  `wavenet_tcn_quality` on clean/crunch when runtime margin matters.

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
2. Export clean balanced, clean quality, crunch balanced, crunch quality, and
   rhythm quality; compare native benchmark matrix results.
3. If rhythm still has audible residual, try the top latency candidates from
   preparation (`2`, `10`, and `18` samples) before another long run.
4. Add a second clean or edge-of-breakup capture to make sure the clean result
   was not specific to this amp setting.
