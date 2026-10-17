# Version 10.1 model reference

[README](../README.md) · [Training and resume command table](../README.md#training-commands) · [V10 reference](model-v10.md)

V10.1 is `onsets-stft-cqt-harmonic-recurrent`, with **3,054,988 trainable parameters** at default settings. Fresh auto training now selects V12.1; use the explicit V10.1 architecture flag for this version. Model version is `10.1`; checkpoint format version is `12`. Resume restores the saved architecture, and explicit V10/V9/V8.1 commands remain available in the README table.

## Structure

```text
16 kHz mono waveform
  ├─ Short STFT 2048 → CNN → 200 features ────────────────┐
  └─ CQT → harmonic stack → CNN → 200 features ──────────┘
       → concatenate 400 / Linear 400→200 / LayerNorm / GELU / dropout
       → three independent residual temporal CNN blocks
       → two-layer bidirectional GRU, 128 units per direction
       → frame, onset, offset, velocity and pedal
```

Harmonic stacking is applied **only to the CQT branch**. The 8192-sample long mel-FFT branch from V10 has been replaced. Both branches use the existing **320-sample / 20 ms hop** and centered constant padding, giving `floor(samples / 320) + 1` output frames. Four seconds produces **201 frames in every head**.

## Frontends

| Branch | Representation | Default window/grid | CNN input |
| --- | --- | --- | --- |
| Short STFT | Full, linearly spaced magnitude spectrum; no mel transform | 2048-sample periodic Hann, 128 ms, 7.8125 Hz spacing | `[batch, 1, 1025, frames]` |
| CQT | V10's fixed variable-length constant-Q filters followed by harmonic stacking | 295 CQT bins, A0 to below 8 kHz, three bins per semitone | `[batch, 8, 264, frames]` |

The STFT magnitude is divided by its fixed Hann window sum (`n_fft / 2`). A sinusoid at an exact FFT bin therefore has a magnitude near half its waveform amplitude, matching the CQT's L1 filter convention. Both branches use `log1p(10 * magnitude)` before the CNN. Waveform amplitude is preserved; no individual audio window is normalized by its measured loudness. Spectral calculations stay in float32 under autocast.

CQT and harmonic stacking retain V10's implementation. Channels align evidence at ratios **0.5, 1, 2, 3, 4, 5, 6 and 7** of candidate frequencies. Missing frequencies are zero. The lowest CQT filter spans approximately 1.87 seconds; high-note filters are shorter. Its octave FFT projections are an implementation of those CQT filters, rather than the removed fixed 8192-sample mel-FFT branch. See [V10 frontend details](model-v10.md#cqt-and-harmonic-stacking).

## CNNs and temporal processing

Each independent CNN uses channels **16, 32, 64 and 96**, 3×3 convolutions, eight-group GroupNorm and GELU. Pooling reduces frequency only:

| Branch | Frequency pooling factors | Positions after pooling |
| --- | --- | ---: |
| Short STFT | 2, 2, 2, 8 | 17 |
| Harmonic CQT | 2, 2, 2, 2 | 17 |

Ceiling division retains the uppermost bins. Extra final STFT pooling keeps the default model near three million parameters while accepting its complete 1025-bin spectrum. The flattened frequency/channel features project to 200 features per frame. Fusion concatenates 400 features and projects to 200.

The remaining structure is shared with V10: three independent residual CNN blocks using temporal `200→400→200` convolutions, kernel 3, padding 1, GroupNorm, GELU and dropout 0.2. The two-layer bidirectional GRU has 128 hidden units per direction and dropout 0.2 between layers. It uses past and future context within each analysis section, so this remains an offline transcription model.

The frame head is conditioned on onset probabilities. Offset refinement is conditioned on sequence features and frame/onset/pedal probabilities. Frame, onset, offset and velocity output 88 logits per frame; pedal outputs one. MIDI key-release labels supervise offsets, while sustain-controller labels supervise pedal separately.

## Parameters and options

| Component | Trainable parameters |
| --- | ---: |
| Short STFT CNN encoder | 406,104 |
| CQT harmonic CNN encoder | 407,112 |
| Fusion projection and normalization | 80,600 |
| Three residual CNN blocks | 1,445,400 |
| Bidirectional GRU | 549,888 |
| Heads, sequence normalization and offset refinement | 165,881 |
| Three global thresholds | 3 |
| **Total** | **3,054,988** |

The STFT, CQT and harmonic stack have no trainable parameters. Defaults are feature width 200, hidden size 128 per direction, two GRU layers, short FFT 2048, release persistence three frames and dropout 0.2.

- `--short-fft` is specific to V10.1; it accepts a power of two at least 512. Changing it changes temporal/frequency resolution and the encoder parameter count.
- `--long-fft` retains its meaning for earlier versions and is rejected for V10.1.
- Feature width must be an integer at least 16 and divisible by eight.
- Existing GRU size/layers, release persistence and recurrence backend options apply.
- Temporal Fourier-mode/layer options remain specific to V8.1.

## Training and compatibility

V10.1 retains the current frame/onset/offset/release/velocity/pedal losses, three gradient-trained global thresholds, equal pitch weights, ordinary random training windows and deterministic validation windows. Best checkpoint selection and scheduling minimize validation **AvgF loss**. Early stopping remains disabled. Comparable frame BCE and weighted loss-component logging remain available.

New files are `piano-v10.1.pt`, `piano-v10.1.last.pt` and `piano-v10.1.history.jsonl`. Resume restores weights, thresholds, optimizer, scheduler, epoch and random state. Fresh runs use random weights. Compatible V10.1 `--init-from` starts a new optimizer/epoch count; initialization from older versions is rejected because the feature structure differs.

V10 retains **3,035,788 parameters** and its original model configuration/state keys. V9 and V8.1 retain **10,168,060** and **10,012,924** parameters. Their explicit start/resume commands are preserved. All README shell commands are now in tables; the training table includes V10.1 through the frame baseline, and retains the previous V8 continuation command.

The desktop app, CLI and local Python web app can use V10.1 `.pt` checkpoints, with the existing overlap methods, score, ByteDance comparison and playback controls. The browser-only Site/exporter remains limited to V8.1/V9.

## Verification

`tests/test_v101.py` checks the default parameter budget, branch input shapes, harmonic stacking on CQT, four-second alignment and finite nonzero gradients for every trainable parameter. It checks STFT frequency bins, fixed amplitude scaling, impulse centers, short/odd inputs, empty event targets and CUDA optimization with a batch of four.

Small training fixtures check default file naming, metadata, save/load/resume, compatible initialization, exact optimizer/scheduler/random-state restoration, minimum AvgF selection, completion of requested epochs, logging and all three overlap modes. README examples are parsed without executing them. Existing V10/V9/V8.1 regression checks retain their structures and loading behavior.

| Check | Command |
| --- | --- |
| V10.1 model and training pipeline | `python -m unittest tests.test_v101 -v` |
| Older models and app selection | `python -m unittest tests.test_v10 tests.test_v9 tests.test_v8 tests.test_app_overlap -v` |

These checks verify execution and compatibility. Full V10.1 training and recognition-quality comparison have not been run.

Verified on **2026-10-12**: **78 targeted checks passed**, including CPU/CUDA default-size optimization, V10.1 resume/evaluation, V10/V9/V8.1 regression, the optimized playback UI, model selection and local-web upload/analysis. All **28 prior README commands** from code blocks were preserved, and every documented piano CLI example parsed successfully without launching training or downloads.
