# Clean, Crunch, Rhythm, Edge, Lead, And Pedal Capture Baseline

Reviewed: 2026-06-25

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
- Keep `wavenet_tcn_high_gain` hidden as a research-only preset until a
  residual/skip/gated long-receptive-field variant replaces the current plain
  sequential tanh stack.
- Treat lead captures as a special review path when latency confidence is low
  or the target is very compressed.
- Keep `wavenet_tcn_fast` as a quick WaveNet probe.
- Keep `conv1d_stack_prelu` as a speed fallback, not the main amp-quality path.

## Second-Generation CLEAN3 Check

Project `project_5549035413254baa9da5197f17e05e5e` is the first full-length
test of the new `DI3.wav` capture workflow. It used the re-exported
`CLEAN3.wav`, known latency, the optimized prep path, WaveNet balanced training,
and a full RTNeural export package.

### Capture And Prep

| Field | Value |
| --- | --- |
| Project name | `2026 CLEAN TEST` |
| Dry input | `DI3.wav` |
| Target | `CLEAN3.wav` |
| Duration | `613.08 s` |
| Format | 48 kHz, stereo dual-mono mixed to mono |
| DI peak / RMS | `-5.56 dBFS` / `-31.50 dBFS` |
| Target peak / RMS | `-5.09 dBFS` / `-23.63 dBFS` |
| RMS delta | `+7.87 dB` |
| Clipping | none |
| Latency | known `13 samples` |
| Prep status | ready |

The capture is healthy. The target has enough headroom, the gain delta is
usable for a clean amp-head render, and the new transient preamble lets us use a
fixed DAW-render latency of `13 samples`. The only prep notes were expected:
stereo was mixed to mono and the capture is long enough that training should
sample windows rather than exhaustively use the whole file every epoch.

### Training Result

| Field | Value |
| --- | --- |
| Run | `run_dfac4dce508e4c438ae25a198a6f7234` |
| Preset | `wavenet_tcn_balanced` |
| Device | `tensorflow-gpu:/physical_device:GPU:0` |
| Requested epochs | `120` |
| Stopped epoch | `66` |
| Best checkpoint epoch | `54` |
| Early stopping reason | `validation_score_plateau` |
| Best composite validation score | `0.00722` |
| Best stream ESR | `0.00439` at epoch `44` |
| Exported preview ESR | `0.01119` |
| Exported preview RMSE | `0.01289` |
| Continuous correlation | `0.99440` |
| Quality verdict | excellent |

The learning-rate schedule behaved correctly. It started at `0.0007`, reduced
five times after validation plateaus, and early-stopped after the composite
validation score failed to beat epoch `54` within patience. This is a good sign:
the scheduler did not run blindly to the requested epoch count, and the best
checkpoint was preserved for export.

The final preview metric is not as low as the older short clean quality run, but
this is a much more varied 10-minute capture and the first pass used balanced,
not quality. The result is still an excellent export candidate, especially
given the correlation and native runtime margin.

### Export Result

| Gate | Result |
| --- | --- |
| RTNeural validation | pass |
| Validation RMSE | `0.00000953` |
| Validation max abs error | `0.00002849` |
| Native benchmark | pass |
| Worst-case native RTF | `20.52x` stereo, block size `64` |
| Model size | `212,020 bytes` |
| Receptive field | `511 samples` / `10.65 ms` |
| Export latency | `13 samples` / `0.27 ms` |
| Aliasing verdict | low aliasing |
| Average ASR | `0.00174` |
| Worst ASR | `0.00499` at ~`5 kHz` probe |

This export is a strong product sanity check for `wavenet_tcn_balanced`: it
passes parity, has comfortable native Eigen headroom, and has low ASR on the
current sine-probe aliasing report. The native benchmark matrix only had the
primary Eigen backend available in this build; STL and xsimd remain listed but
not measured.

### Interpretation

The CLEAN3 run validates three newer workflow changes:

- Known latency is the right path for DAW-rendered captures that share one DI
  and render chain. It avoids heavy-tone guesswork and gives deterministic prep.
- Long, varied captures are usable. They train more slowly, but the
  energy-stratified window sampler selected `3,685` windows from `7,183`
  available windows and held validation windows fixed while resampling training
  windows each epoch.
- WaveNet balanced is strong enough for clean production captures, not just high
  gain. Quality may still win a metric shootout, but balanced exported with far
  more native headroom than the minimum needed for real-time use.

## Second-Generation RHYTHM3B Check

Project `project_f94c77d3aefe4f5e8abbaf3a86cfcf6a` is the first successful
second-generation heavy rhythm result after the long RHYTHM3 capture struggled
with balanced WaveNet. It used the trimmed `DI3-B_1.wav` / `RHYTHM3-B.wav`
pair, float32 prepared WAVs, the `wavenet_tcn_quality` preset, and export
package `export_0459ae977d3e4e38b891718b94ec3305`.

### Capture And Prep

| Field | Value |
| --- | --- |
| Project name | `Test Rhythm 3B` |
| Dry input | `DI3-B_1.wav` |
| Target | `RHYTHM3-B.wav` |
| Duration | `444.25 s` |
| Format | 48 kHz, stereo dual-mono mixed to mono |
| DI peak / RMS | `-5.44 dBFS` / `-29.86 dBFS` |
| Target peak / RMS | `-9.49 dBFS` / `-20.25 dBFS` |
| RMS delta | `+9.62 dB` |
| Clipping | none |
| Prepared sample format | float32 WAV |
| Latency | estimated `10 samples` |
| Latency confidence / agreement | `0.40` / `42%` |
| Top candidates | `10`, `399`, `-478`, `2`, `301` samples |

The capture is technically healthy: no clipping, reasonable headroom, and a
target RMS that makes sense for a dense raw amp-head rhythm tone. The latency
estimate remains weak, though. The preamble and trimming helped, but they did
not make this high-gain rhythm tone an easy alignment case.

### Training Result

| Field | Value |
| --- | --- |
| Run | `run_9a920dd9be4347369519547ada5d9395` |
| Preset | `wavenet_tcn_quality` |
| Device | `tensorflow-gpu:/physical_device:GPU:0` |
| Requested epochs | `120` |
| Best checkpoint epoch | `119` |
| Loss | `mrstft_preemphasis` |
| Preview/state-continuous ESR | `0.11670` |
| Stream validation ESR | `0.11075` |
| Window validation ESR | `0.10639` |
| RMSE | `0.03575` |
| Continuous correlation | `0.94049` |
| Quality verdict | good export candidate |

This run is the important correction to the earlier RHYTHM3B balanced result.
Balanced underfit badly on this capture, even after trimming silence and
preserving float32 prep. Quality mode crossed into useful territory and was
still improving late in the run, with the best checkpoint at epoch `119`.

### Export Result

| Gate | Result |
| --- | --- |
| Export package | `export_0459ae977d3e4e38b891718b94ec3305` |
| RTNeural validation | pass |
| Validation RMSE | `0.00000934` |
| Validation max abs error | `0.00002556` |
| Native benchmark | pass |
| Worst-case native RTF | `11.78x` stereo, block size `128` |
| Model size | `413,571 bytes` |
| Receptive field | `2047 samples` / `42.65 ms` |
| Export latency | `10 samples` / `0.21 ms` |
| Aliasing verdict | review aliasing |
| Average ASR | `0.02516` |
| Worst ASR | `0.06779` at ~`5 kHz` probe |

The export is compatible and fast enough for native testing. The aliasing
warning is the main caveat. The `1.25 kHz` and `2.5 kHz` probes were low, while
the ~`5 kHz` probe landed in the review band. A follow-up amplitude sweep from
`0.05` to `0.75` input amplitude kept worst ASR around `0.051-0.070`, so this is
not merely the default sine probe driving the model too hard.

### Interpretation

RHYTHM3B confirms that raw high-gain rhythm should default to
`wavenet_tcn_quality` when the user wants production quality. Balanced is still
the correct first run for many profiles, but this capture shows a real capacity
floor: silence trimming, float32 prep, and better DI material were not enough
for balanced to model the upper saturation/fizz region.

The ASR warning should be handled by listening, not by rejecting the export.
Because the capture is an amp head without a cabinet, upper harmonic/fizz
content is exposed. Listen for metallic foldback on sustained high notes,
natural harmonics, and high-register bends. If the model sounds good in those
cases, the export is a valid test candidate; if the warning is audible, the next
product-side experiment is plugin oversampling or a higher-sample-rate export
path.

## Short RHYTHM4 Check

Project `project_ab40008405d546398afff4a8d6a8dde7` used the shorter
`DI4.wav` / `RHYTHM4.wav` pair. It was 158.8 seconds long, 48 kHz stereo
dual-mono prepared as mono float32, with input peak `-6.30 dBFS`, target peak
`-10.59 dBFS`, target RMS `-21.23 dBFS`, and no clipping. The effective
latency estimate was `9 samples`, but confidence stayed low at `0.41`, so
alignment still deserves review on this capture family.

| Preset | ESR | RMSE | Correlation | Epochs | Best Epoch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_balanced` | `0.6309` | `0.0740` | `0.6109` | `14` | `2` | Plateaued immediately; too small for this rhythm tone. |
| `wavenet_tcn_quality` | `0.1369` | `0.0345` | `0.9295` | `120` | `120` | Continued improving and exported successfully. |
| `wavenet_tcn_high_gain` | `0.6310` | `0.0740` | `0.6142` | `37` | `13` | Underperformed quality; hidden from normal recommendations. |

This reinforces two lessons. First, captures do not need to be extremely long:
roughly 2.5-4 minutes can be enough if the performance is varied and trimmed.
Second, dense raw amp-head rhythm still exposes a capacity/optimization floor
in `wavenet_tcn_balanced`. The longer `wavenet_tcn_high_gain` variant did not
fix that floor: it matched the failed balanced result instead of the successful
quality result. Preview analysis showed no latency offset and no simple gain
fix; the hidden Conv1D activations stayed tiny, which points to optimization
collapse in the deeper non-residual tanh stack.

Continuation update: the later `wavenet_tcn_quality` run
`run_1fabc58f146a47a5bc9ac9a47e5b7592` pushed this same RHYTHM4 capture to
test ESR `0.0713`, correlation `0.9640`, and native validation max abs error
`2.35e-5`. The preview was strong, but the residual stayed most visible in the
upper bands and the export raised an ASR warning. That makes
`wavenet_tcn_quality_tanh15` the next safe architecture probe: it keeps the
successful quality receptive field and RTNeural layer graph while testing a
gentler training tanh before considering larger residual/gated WaveNet work.

Smoothed-tanh update: `wavenet_tcn_quality_tanh15` has now beaten the plain
quality chain on this same RHYTHM4 capture. Run
`run_cc3dc9235cf7426b8529c546003e0e75` resumed from the previous tanh15
checkpoint, requested `180` more epochs, stopped at epoch `691` on validation
score plateau, and selected epoch `671`. Its final preview metrics were ESR
`0.0646`, RMSE `0.0237`, correlation `0.9674`, and residual RMS
`-32.51 dBFS`. The associated export
`export_cc5a4ffee7b6400c99476acf7967feeb` passed native validation
(`2.73e-5` max abs error) and benchmarked at `11.74x` worst-case realtime on
the Eigen backend. Compared with the previous plain `wavenet_tcn_quality`
export, this improved ESR by about `9.4%`, lowered worst ASR from `0.290` to
`0.067`, and lowered average ASR from `0.100` to `0.042`. The ASR report still
warns, so listening for foldback on high notes remains necessary, but this is
the first smoothed-tanh result here that improves both waveform fit and aliasing
risk against the same quality baseline.

A2-inspired update: `wavenet_tcn_a2_prelu` beat the full tanh15 continuation
chain in one fresh 180-epoch run. Run
`run_e61c249debfa4f04a140cf0ff9d7f4ff` selected its best checkpoint at the end
of training, with no early-stop plateau. Its export
`export_59748d4bbcf24f97bd414fc4b3365699` passed native RTNeural validation
(`2.23e-5` max abs error) and benchmarked at `6.54x` worst-case realtime on the
Eigen backend. The package metrics were ESR `0.0440`, MAE `0.0104`, RMSE
`0.0196`, and state-continuous correlation `0.9778`. Compared with the best
quality-tanh15 export, this improved ESR by about `32%`, average ASR by about
`51%` (`0.0419` to `0.0205`), and worst ASR by about `47%` (`0.0670` to
`0.0354`). The trade-off is runtime and size: A2 PReLU uses 12 Conv1D/PReLU
blocks, a 2,481-sample receptive field, and an `832 KB` JSON, versus tanh15's
10 Conv1D blocks, 2,047-sample receptive field, `416 KB` JSON, and `11.74x`
worst-case realtime. It is still plugin-ready on this machine, but no longer
belongs in the "just research" bucket for high-gain captures.

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

Follow-up rhythm2 smoothed-tanh rerun, after adding ASR export diagnostics:

| Run | Preset | Preview ESR | RMSE | Corr | Residual RMS | Est. RTF | Stream Val Score | Worst ASR | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `run_2fae65f91e754b0c942ffeed3cc4b0f2` | `wavenet_tcn_balanced` | `0.1463` | `0.0610` | `0.9245` | `-24.29 dBFS` | `3.0x` | `0.3320` | `0.1463` | Best waveform metrics in this rerun. |
| `run_43464bcdeb00499d8f6b15154c920aaa` | `wavenet_tcn_balanced_tanh15` | `0.1873` | `0.0690` | `0.9025` | `-23.22 dBFS` | `3.0x` | `0.4082` | `0.4302` | Better of the smoothed-tanh quality runs, but higher ASR. |
| `run_505fd2d5a4a948baaf8cbd3a733fac93` | `wavenet_tcn_balanced_tanh18` | `0.2058` | `0.0724` | `0.8934` | `-22.81 dBFS` | `3.0x` | `0.4269` | `0.1104` | Weaker waveform metrics, but best ASR. |

Important correction: the first UI/report pass showed the smoothed-tanh presets
at `120x` estimated RTF. That was a trainer estimate bug caused by the new
preset IDs falling through to the generic Conv1D tier. The exported graphs have
the same layer shape as `wavenet_tcn_balanced`; after fixing the estimator, they
correctly report the balanced-class `3.0x` estimate. Temporary exports for all
three runs passed RTNeural JSON parity with max absolute error below `0.00001`.

This is still useful evidence. `tanh15` sounds/metrics-wise sits between fast
and balanced, while `tanh18` behaves more like an anti-aliasing probe: worse
ESR/RMSE than `tanh15`, but lower ASR than even the balanced baseline. The
smoothed activations should stay in research status until listening tests tell
us whether the lower-ASR trade-off is audible on sustained high notes.

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
| `wavenet_tcn_separable_fast` | `0.0833` | `0.0561` | `0.9577` | `-25.02 dBFS` | `5.0x` | `120` | New grouped/dilated experiment; quality is on par. |
| `conv1d_stack_prelu` | `0.1798` | `0.0824` | `0.9064` | `-21.68 dBFS` | `120x` | `120` | Needs work. |
| `wavenet_tcn_fast` | `0.1969` | `0.0863` | `0.8989` | `-21.28 dBFS` | `8.0x` | `120` | Needs work. |

Lead is the first capture where quality does not beat balanced. The new
`wavenet_tcn_separable_fast` preset is also effectively tied on metrics, which
makes it a useful research preset for this dense lead tone. It trained on CPU by
design because TensorFlow Metal does not reliably execute grouped Conv1D on this
machine; export parity now honors the saved CPU checkpoint device. A native
Eigen benchmark of the trained separable export produced about `8.90x`
worst-case RTF with a much smaller JSON model (`96 KB`), but balanced still
benchmarked faster at about `18.35x` worst-case RTF. Keep separable as a
static/fused/plugin-side experiment, not the default training recommendation.

The target is very dense: `-14.95 dBFS` RMS with only about `5.2 dB` crest factor. Latency
confidence is low, but a preview shift search still selected shift `0`, so the
rendered residual is real model mismatch rather than a report alignment artifact.
The residual is broad-band and concentrated in the presence/fizz region
(`1-9 kHz`). Since the capture is pure amp head with no cabinet or time-based
effects, this points to dense lead saturation plus latency ambiguity rather than
a bad WAV file or a contaminated target chain.

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
   balanced slightly beat quality and separable-fast landed in the same quality
   range.

2. Stacked Conv is currently a speed fallback.

   `conv1d_stack_prelu` is extremely fast, and it can produce good results on
   simpler captures, but it missed too much behavior on every amp capture in
   this set and trailed WaveNet on the overdrive pedal capture. It should not be
   the default quality recommendation unless future capture families prove a
   repeatable exception.

3. Rhythm needs more than one 120-epoch quality pass.

   `wavenet_tcn_quality` remained the best rhythm model and did not early-stop.
   Continue-from-best with a lower learning rate is justified, especially before
   comparing exports. The rhythm2 smoothed-tanh rerun also shows that activation
   smoothing creates a real ESR-versus-ASR trade-off rather than a simple upgrade.

4. Lead needs a special review path.

   Lead had the hottest RMS, lowest crest factor, and weakest latency confidence
   in this set. Balanced and quality tied, and the residual stayed broad-band in
   the presence/fizz region. The capture is already pure amp head, so future
   lead work should focus on transient pre-roll, top latency candidates, and
   balanced-versus-quality comparisons rather than removing nonexistent
   downstream effects.

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
4. If lead remains harder, re-test with the transient pre-roll and candidate
   latency offsets before another long production run.
5. If rhythm still has audible residual, try the top latency candidates from
   preparation (`2`, `10`, and `18` samples) before another long run.
6. CLEAN3 confirms the second clean capture path. Next, run the same DI3
   workflow on edge, crunch, rhythm, and lead with known latency set to
   `13 samples`.
7. For CLEAN3 specifically, compare `wavenet_tcn_quality` and one fast fallback
   against the balanced export only if listening reveals a gap; the balanced
   export already has excellent quality and comfortable native headroom.
