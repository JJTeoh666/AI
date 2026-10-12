# Version 9 model reference

[Back to the README](../README.md) · [V9 training commands](../README.md#training-commands) · [V8.1 reference](model-v8.md)

V9 is a separate architecture, `onsets-triple-residual-recurrent`, with **10,168,060 trainable parameters** at default settings. Use its explicit architecture flag to start V9; fresh auto training now selects V10.1. Model version is `9`; checkpoint format version is `10`. Resume restores the saved architecture, so existing V8/V8.1 runs continue as V8.1.

## Audio branches

```text
16 kHz mono waveform
  ├─ FFT 2048 → 128 mel bands → independent CNN → 384 features ─┐
  ├─ FFT 8192 → 256 mel bands → independent CNN → 384 features ─┼─ concatenate 1152
  └─ raw Hann windows 2048 → FAN → LayerNorm → CNN → 384 ──────┘
       → Linear 1152→384 / LayerNorm / GELU / dropout
       → three independent residual temporal CNN blocks
       → two-layer bidirectional GRU, 192 units per direction
       → LayerNorm / frame, onset, offset, velocity and pedal heads
```

The FFT branches reuse V8.1's frontends and encoder classes. Each CNN has channels 16, 32, 64 and 128, 3×3 convolutions, eight-group GroupNorm and GELU. Pooling only the frequency dimension preserves frame alignment. Flattening frequency positions before projection preserves absolute pitch information. Mel coverage is 20–8000 Hz.

Every branch uses a **320-sample / 20 ms hop** and centered constant padding. Raw windows pad 1024 samples on each side before extracting 2048-sample windows. Centers occur at samples 0, 320, 640, and so on; multiplying by a registered Hann buffer preserves waveform amplitude. There is no per-window amplitude normalization. For `N` input samples, each branch produces `floor(N / 320) + 1` frames. A four-second input produces 201 frames, including the endpoint that inference handles when joining sections.

The raw FAN uses the existing learned projection:

```text
concat(cos(Wp x), sin(Wp x), GELU(Wa x + ba))
```

At width 384, this gives 96 cosine, 96 sine and 192 nonperiodic features. It follows the [authors' FAN reference](https://github.com/YihongDong/FAN). After LayerNorm, a temporal Conv1d `384→384`, kernel 3, padding 1, GroupNorm, GELU and dropout process neighbouring frames.

## Residual convolution and recurrence

Each of the three blocks has its own weights:

```text
x → Conv1d 384→768 / GroupNorm(8) / GELU / dropout
  → Conv1d 768→384 / GroupNorm(8) / dropout
  → add x directly → GELU
```

Both convolutions use kernel size 3 and padding 1. There is no learned skip scale. Fusion and convolution dropout are 0.2. The two-layer bidirectional GRU uses 192 hidden units per direction and dropout 0.2 between layers. `PortableGRU` retains the existing native/CUDA backend behavior, including the Windows nightly fallback.

Bidirectional recurrence can use preceding and following audio within the analysis section. This remains an offline transcription model. Inference offers Keep center, Weighted average and 50/50 average using the same predictions and decoding as earlier versions.

## Outputs and training

Frame, onset, offset and velocity heads each output 88 logits per frame; pedal outputs one logit. Onset probabilities condition the frame head. Frame, onset and pedal probabilities condition the additive offset refinement network. MIDI key releases supervise offsets; pedal controller labels supervise sustain separately. The decoder uses the pedal prediction to represent sustain independently of physical key release.

V9 retains V8.1's objective and validation behavior:

- Frame, onset, offset, release, velocity and pedal losses, plus the weighted threshold objective.
- Three gradient-trained global thresholds for frame, onset and offset; smooth mean pitch F0/F1/F2/F3 objective.
- Equal pitch weights across all 88 keys and ordinary random training windows by default.
- The same deterministic validation windows and note-boundary exclusions.
- Minimum validation `avgf_loss` for best checkpoints and learning-rate scheduling; improvement must exceed `1e-6`.
- No early stopping; all requested epochs run. Comparable frame BCE and weighted component reporting remain available.

Best/latest files are `piano-v9.pt` and `piano-v9.last.pt`; history is `piano-v9.history.jsonl`. Version metadata appears in checkpoints, history, evaluation, transcription and the app. Random initialization is the default. Compatible V9 `--init-from` transfers recognizer weights and restarts optimizer/epoch state. V8.1-to-V9 initialization is rejected because the structures differ.

## Parameter budget and options

| Component | Trainable parameters |
| --- | ---: |
| Short and long FFT CNN encoders | 1,770,432 |
| Raw FAN, normalization and temporal CNN | 1,034,304 |
| Fusion projection and normalization | 443,520 |
| Three residual CNN blocks | 5,318,784 |
| Bidirectional GRU | 1,331,712 |
| Output heads, sequence normalization and offset refinement | 269,305 |
| Global threshold parameters | 3 |
| **Total** | **10,168,060** |

The count uses feature width 384, hidden size 192, two GRU layers and long FFT 8192. Existing `--feature-width`, `--hidden-size`, `--gru-layers`, `--long-fft`, `--release-frames` and `--rnn-backend` options apply. V9 feature width must be at least 16 and divisible by eight. Three residual blocks and the 2048-sample raw window are fixed. Temporal `--fourier-modes` and `--fourier-layers` are exclusive to V8.1.

## Verification

`tests/test_v9.py` checks the default parameter budget and a four-second forward/backward pass, including finite nonzero gradients for every trainable parameter. It also checks short and odd-length frame alignment, raw-window amplitude/padding, independent direct residual paths, tiny training runs, save/load/resume, initialization restrictions, AvgF selection, scheduling, loss reporting, full-recording evaluation and all three overlap modes. The app tests check V9 preference and version display. Existing V8.1 tests retain its 10,012,924-parameter structure, explicit commands and checkpoint compatibility.

These checks verify execution and compatibility. Full V9 training and recognition-quality comparison are separate work.

Verified on 2026-10-08: **48 tests passed** in 22.5 seconds, including the default-size four-second forward/backward check on CPU and exact uninterrupted-versus-resumed model, optimizer, scheduler and random-state comparisons. The app checks ran with a hidden Tk window.

```powershell
python -m unittest tests.test_v9 tests.test_v8 tests.test_avgf tests.test_loss_reporting tests.test_app_overlap tests.test_inference_overlap -v
```
