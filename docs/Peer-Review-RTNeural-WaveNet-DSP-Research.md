# Peer Review: RTNeural WaveNet DSP Research Paper

Date: 2026-06-29  
Paper reviewed: [Paper-RTNeural-WaveNet-DSP-Research.md](Paper-RTNeural-WaveNet-DSP-Research.md)

## Review Scope

This peer review evaluated the paper as an internal scientific DSP/audio-ML
report, not as product marketing. The review focused on overclaims, missing
methodology, metric definitions, runtime caveats, causal interpretation, and
whether the conclusion followed from the available evidence.

## Reviewer Summary

The paper credibly argues that WaveNet-family causal TCN presets are the
best-tested product path in the current RTNeural-constrained workflow. However,
the first draft supported a narrower claim than it initially made. Several
statements read as general architecture conclusions even though the evidence is
from internal product experiments, often single-seed runs, and not matched
recurrent-versus-TCN ablations.

The reviewer recommended narrowing the thesis, making the method more
reproducible, clearly separating observed results from causal explanations, and
scoping runtime claims to the reference system.

## Major Review Findings

1. Architecture claims were too broad.

   The original draft implied that finite-memory causal Conv1D models are
   generally better than LSTM/GRU or shallow Conv models. The evidence is
   strongest for the current internal 48 kHz mono paired-capture workflow and
   RTNeural JSON constraints. It does not establish a general architecture
   theorem.

2. Several causal interpretations were not isolated.

   The RHYTHM4 A2 result and real-hardware capture success are compelling, but
   they are confounded by capture workflow maturity, training continuation,
   model capacity, preparation changes, target type, and capture length. These
   should be presented as hypotheses or associated observations unless future
   ablations isolate them.

3. The method section needed more reproducibility detail.

   The first draft described the pipeline well but omitted several
   implementation specifics: train/validation metric taxonomy, optimizer,
   default seed, batch size, loss weights, MR-STFT frame sizes, LR plateau
   behavior, ASR amplitude/window/warmup, and benchmark context.

4. Runtime conclusions needed tighter scope.

   Native RTF and Logic/AU smoke tests are useful evidence, but they were mostly
   collected on a powerful Apple Silicon reference machine with a debug plugin.
   The paper should not imply broad production readiness across weaker machines
   or release builds.

## Minor Review Findings

- ASR should be described as a probe-specific engineering diagnostic, not an
  audibility score or aliasing ground truth.
- RMSE and residual RMS were redundant without explanation.
- Prediction RMS ratio appeared before being defined.
- Tone labels such as clean and edge-of-breakup need context because pickup
  type and playing level can move a capture between categories.
- References are internal-doc and URL based; formal publication formatting can
  be added later if this becomes an external paper.

## Author Response And Revisions

The paper was revised after review.

1. Thesis narrowed.

   The abstract, discussion, and conclusion now state that WaveNet-family
   causal Conv1D presets are the best-tested product path within the current
   internal 48 kHz mono paired-capture dataset and RTNeural JSON constraints.
   Broad claims about general superiority over all recurrent or convolutional
   alternatives were removed.

2. Methods strengthened.

   A reproducibility envelope table was added covering audio format, pairing,
   latency, polarity, sequence length, window budget, default seed, batch size,
   optimizer, early stopping, LR plateau schedule, pre-emphasis coefficients,
   MR-STFT weights/frame sizes, ASR settings, native benchmark context, and
   plugin smoke-test context.

3. Metric taxonomy added.

   The paper now defines window validation, stream validation, composite
   validation score, preview/state-continuous metrics, prediction RMS ratio, and
   residual RMS in relation to RMSE.

4. Architecture hyperparameters added.

   The WaveNet preset table now lists layers, filters, kernels, dilations,
   activations, losses, and intended use.

5. Causal explanations softened.

   A2 PReLU improvements are now described as associated with the combined
   architecture changes, not as proof of which individual feature caused the
   improvement. Hardware-capture advantages are framed as plausible hypotheses
   for future matched experiments.

6. Runtime scope tightened.

   Runtime claims now say "reference system" and "debug-plugin smoke path"
   where appropriate. Future work still calls for pluginval, weaker-machine
   testing, release builds, and systematic CPU profiling.

## Remaining Scientific Limitations

The revised paper is suitable as an internal scientific report, but not yet as a
fully controlled external publication. Before external publication, the project
should add:

- Repeated-seed runs for key presets.
- Matched train/validation/test splits across architecture comparisons.
- Controlled latency-offset ablations for low-confidence captures.
- A blinded listening study for target/prediction/residual comparisons.
- Listening-calibrated ASR thresholds.
- Controlled native/plugin benchmarks across weaker machines, sample rates,
  block sizes, and pedal + amp + IR chains.
- Formal bibliography formatting.

## Final Review Decision

Accept as an internal technical paper after revision. For external publication,
the paper would need additional controlled experiments and formal citation
formatting.
