# Version 10 model reference

[Back to the README](../README.md) · [V10 training commands](../README.md#training-commands) · [V9 commands](../README.md#training-commands) · [V8.1 commands](../README.md#training-commands)

V10 is `onsets-cqt-harmonic-recurrent`, with **3,035,788 trainable parameters** at the default dimensions. Fresh auto training now selects V12.1; use the explicit architecture flag to start V10. Its model version is `10` and checkpoint format version is `11`. Resume restores the checkpoint's architecture; the explicit V9 and V8.1 training commands remain available and their structures are unchanged.

## Structure

```text
16 kHz mono waveform
  ├─ CQT → harmonic stack → CNN → 200 features ───────┐
  └─ FFT 8192 → 256 mel bands → CNN → 200 features ──┘
       → concatenate 400 / Linear 400→200 / LayerNorm / GELU / dropout
       → three independent residual temporal CNN blocks
       → two-layer bidirectional GRU, 128 units per direction
       → LayerNorm / frame, onset, offset, velocity and pedal heads
```

Both branches use a **320-sample / 20 ms hop**, constant zero padding and frame centers at samples 0, 320, 640, etc. `N` samples produce `floor(N / 320) + 1` frames. Four seconds produces **201 frames for every head**. Inference removes duplicated endpoint frames using the existing stitching/averaging code.

## CQT and harmonic stacking

The frontend uses **295 fixed CQT bins**, starting at A0, 27.5 Hz, and ending below the 8 kHz Nyquist limit. There are **36 bins per octave**, or three per semitone:

```text
f[k] = 27.5 * 2**(k / 36)
Q = 1 / (2**(1 / 36) - 1)
filter_length[k] = ceil(Q * 16000 / f[k])
```

Each complex sinusoid has its own Hann window, normalized by the window's sum. Low-frequency filters therefore cover longer durations than high-frequency filters. The lowest filter covers approximately **1.87 seconds**; FFT groups use sizes 32768 down to 128. All groups keep the same hop, without octave downsampling or interpolation of timestamps. Waveform amplitude is preserved; no audio window is independently normalized.

To keep computation practical, bins in each octave share a rectangular FFT. A Parseval projection converts the FFT coefficients into the real and imaginary responses of the variable-length Hann wavelets. Frequency coefficients smaller than `1e-5` of a wavelet's spectral peak are omitted. Tests compare this approximation against direct, full time-domain filter projections. Registered kernels are fixed buffers, reconstructed when loading a model; they add no trainable parameters and require no additional package. FFT calculations stay in float32 under autocast.

Eight channels align energy at **0.5, 1, 2, 3, 4, 5, 6 and 7 times** each candidate frequency. For candidate bin `k` and harmonic ratio `h`:

```text
shift[h] = round(36 * log2(h))
stack[h, k, t] = CQT[k + shift[h], t]
```

Unavailable frequencies below A0 or above the last CQT bin are zero. The stack retains **264 candidate bins**, three for each of the 88 piano keys, giving `[batch, 8, 264, time]`. Extra upper CQT bins provide observable harmonics of lower fundamentals. Stacking organizes evidence; the CNN must still learn to distinguish harmonics from separately played notes. The CQT representation uses `log1p(10 * magnitude)` before its CNN; the long branch retains `log1p(mel_power)`.

The frequency layout and harmonic ratios follow the approach described in the [Basic Pitch paper](https://arxiv.org/abs/2203.09893) and [Spotify's harmonic stacking implementation](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/nn.py). The constant-Q wavelet length formula is also described in [nnAudio's CQT reference implementation](https://github.com/KinWaiCheuk/nnAudio/blob/master/Installation/nnAudio/features/cqt.py). This project's FFT projection frontend is implemented locally in PyTorch.

## CNNs, fusion and recurrence

Both independent branch CNNs use channels **16, 32, 64, 96**, 3×3 convolutions, eight-group GroupNorm and GELU. Each stage pools frequency only, with ceiling division so the highest candidate bins survive. The CQT branch retains 17 frequency positions after pooling; the 256-band long FFT branch retains 16. Flattening channels and frequency before projection preserves absolute pitch information. Each branch projects to 200 features per frame with LayerNorm, GELU and dropout 0.2.

Fusion concatenates the branches and projects `400→200`. Each independent residual block performs:

```text
x → Conv1d 200→400 / GroupNorm(8) / GELU / dropout
  → Conv1d 400→200 / GroupNorm(8) / dropout
  → add x directly → GELU
```

Temporal convolutions use kernel 3, padding 1. The two-layer bidirectional GRU uses 128 hidden units per direction and dropout 0.2 between layers. It has access to earlier and later audio in the analysis section, so V10 remains an offline model. Existing native/CUDA backend choices and the Windows nightly fallback apply.

## Outputs, training and compatibility

Frame, onset, offset and velocity each output 88 logits per frame; pedal outputs one. Onset probabilities condition frames. Frame, onset and pedal probabilities feed the additive offset refinement network. MIDI key-release events supervise offsets; sustain-controller labels supervise pedal separately.

V10 retains:

- Frame, onset, offset, release, velocity and pedal supervision plus the weighted threshold objective.
- Three gradient-trained global frame/onset/offset thresholds, using the smooth mean pitch F0/F1/F2/F3 objective.
- Equal weights for every piano key and ordinary random windows by default.
- The existing deterministic validation windows and clip-boundary handling.
- Minimum validation `avgf_loss` for best checkpoints and learning-rate scheduling; improvement must exceed `1e-6`.
- Disabled early stopping, comparable frame BCE and separate weighted loss-component reporting.

Default outputs are `piano-v10.pt`, `piano-v10.last.pt` and `piano-v10.history.jsonl`. A fresh model uses random weights. `--init-from` accepts compatible V10 checkpoints and starts a fresh optimizer and epoch count. Older architectures cannot initialize V10. Use their own explicit start commands or resume their saved checkpoints.

The desktop app displays V10 and supports the existing score, piano playback, ByteDance comparison and all three overlap methods. The CLI and local Python web app can also use V10 `.pt` files. The **browser-only Site and ONNX exporter currently support V8.1/V9**, without V10's new CQT preprocessing.

## Parameter budget and options

| Component | Trainable parameters |
| --- | ---: |
| CQT harmonic CNN encoder | 407,112 |
| Long FFT CNN encoder | 386,904 |
| Fusion projection and normalization | 80,600 |
| Three residual CNN blocks | 1,445,400 |
| Bidirectional GRU | 549,888 |
| Output heads, sequence normalization and offset refinement | 165,881 |
| Global thresholds | 3 |
| **Total** | **3,035,788** |

The CQT and harmonic stack have zero trainable parameters. These counts assume feature width 200, hidden size 128, two GRU layers and long FFT 8192. `--feature-width` must be an integer at least 16 and divisible by eight. Existing `--hidden-size`, `--gru-layers`, `--long-fft`, `--release-frames` and `--rnn-backend` options apply. CQT spacing, harmonic ratios and three residual blocks are fixed for V10. Temporal Fourier-mode/layer options remain specific to V8.1.

Fewer parameters reduce model-weight storage and neural-network cost. The CQT still has a computation cost of its own; total training speed should be measured on the actual device. Neither the representation nor the smaller model establishes a recognition-quality improvement before training and evaluation.

## Verification

`tests/test_v10.py` checks:

- CQT FFT projections against direct Hann wavelets, tone pitch positions, amplitude scaling, silence and shared impulse centers.
- Harmonic channel shifts, out-of-range masking and retention of the highest piano bins.
- The default parameter count and four-second forward/backward pass, with finite nonzero gradients for every trainable parameter.
- Short and odd-length audio, silence with empty event targets and configuration checks.
- A default-size CUDA batch of four and an optimizer step when CUDA is available; float32 CQT under autocast and CPU/GPU agreement.
- Tiny training runs, save/load/resume, exact optimizer/scheduler/random-state restoration and initialization restrictions.
- Minimum AvgF selection, completion of all requested epochs, loss reporting and full-recording evaluation.
- Keep center, weighted average and 50/50 overlap inference, partial final sections and cancellation.

The existing V8.1/V9 tests retain their 10,012,924/10,168,060 parameter counts, explicit training commands and checkpoint compatibility. App tests check V10 selection/version display alongside existing playback and comparison behavior.

Verified on **2026-10-10**: **69 checks passed** across V10, V8.1/V9 regression, validation/loss reporting, overlap inference, desktop app and V10 local-web upload/analysis. This includes a default-size four-second CPU backward pass, a CUDA batch of four with an optimizer step, and default-dimension V10 training/save/resume on the small fixture dataset. No long training run was launched.

```powershell
python -m unittest tests.test_v10 tests.test_v9 tests.test_v8 tests.test_avgf tests.test_loss_reporting tests.test_app_overlap tests.test_app_comparison tests.test_inference_overlap -v
python -m unittest tests.test_web.WebAppTest.test_v10_checkpoint_upload_metadata_and_analysis -v
```

These are execution and compatibility checks with small training fixtures. Full V10 training and recognition-quality comparison remain separate work.
