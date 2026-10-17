# Version 11 model reference

[README](../README.md) · [Training command table](../README.md#training-commands) · [V10.1 reference](model-v101.md)

V11 is `onsets-stft-fan-sustain-recurrent`, with **10,256,830 trainable parameters** at default settings. Select its explicit architecture flag for a fresh V11 run; V12.1 is now the fresh auto default. Model version is `11`; checkpoint format version is `13`. Existing checkpoints restore their saved architectures, and previous start/resume commands remain in the README table.

## Structure

```text
16 kHz mono audio
  ├─ Short STFT 2048 → CNN → 384 features ───────┐
  └─ Raw Hann windows 8192 → FAN → CNN → 384 ───┘
       → concatenate 768 → Linear 768→384 / LayerNorm / GELU / dropout
            ├─ Four-layer bidirectional GRU, 224 units per direction
            │    → LayerNorm → one residual CNN block, 448→896→448
            │    → frame / onset / offset / velocity
            └─ Independent one-layer bidirectional pedal GRU, 64 per direction
                 → LayerNorm → pedal activity / pedal onset / pedal offset
```

The main residual CNN follows all four recurrent layers. This order differs from V9–V10.1. The lightweight pedal branch reads the fused acoustic features and provides a separate recurrent representation for sustain control.

Frame, onset, offset and velocity each produce **88 logits** per frame, covering MIDI 21–108. The three pedal heads each produce **one logit** per frame. Onset probabilities condition the frame head. Sequence features and frame/onset/pedal probabilities condition offset refinement.

## Input windows and frame alignment

| Branch | Default input context | Processing |
| --- | --- | --- |
| Short STFT | **2048 samples / 128 ms** | Centered periodic Hann, full 1025-bin magnitude spectrum, CNN |
| Raw FAN | **8192 samples / 512 ms** | Centered periodic Hann raw samples, learned cosine/sine/aperiodic projection, LayerNorm, temporal CNN |

Both branches use **320 samples / 20 ms** between frame centers and constant zero padding at recording boundaries. They produce `floor(samples / 320) + 1` aligned frames: **four seconds produces 201 frames in every head**. The FAN receives the same default time span as the previous 8192-sample long FFT, although it learns a different representation. V11 contains no long FFT, CQT or harmonic stacking branch.

The STFT is divided by its fixed Hann-window sum (`short_fft / 2`) and uses `log1p(10 * magnitude)` before its CNN. Its CNN has channels 16, 32, 64 and 96, eight-group GroupNorm, GELU and frequency pooling factors 2, 2, 2 and 8. Ceiling pooling retains the highest frequency bins and leaves 17 positions at default settings.

The raw branch multiplies each waveform window by Hann coefficients, then directly applies the FAN. Its 384 output features contain 96 cosine features, 96 sine features and 192 GELU features. LayerNorm and a temporal convolution follow the FAN. These learned periodic features are not fixed frequency bins measured in Hz.

Neither branch rescales each input window according to its measured loudness. Waveform amplitude is preserved at the input; normalization inside the network is learned feature processing. Centered windows and bidirectional recurrence use later sound as well as earlier sound, so this is an offline model. The default **training window remains four seconds**; the 512 ms FAN context is the local input span for each frame, not the entire recurrent context.

## Sustain targets and playback

V11 changes note-duration semantics from `key_release` to **`sustain_extended`**, similarly to the ByteDance comparison model:

- A key release while sustain is held extends its frame target and note offset to pedal release.
- A later strike of the same pitch caps the earlier note, preserving repeated notes.
- Releases outside a pedal hold retain their key-release endings.
- Pedal activity, pedal-down events and pedal-up events receive separate supervision.

Extension is computed from the complete MIDI performance before selecting training or validation windows. Frames, offset targets and reference note durations therefore use the same sustained interpretation. Note onsets and velocities keep their original meaning.

Results and checkpoints identify these semantics through `note_duration="sustain_extended"`. The staff uses the resulting sustained durations. Playback uses already-extended note ends directly instead of applying the predicted pedal intervals a second time. Older results retain their existing key-release/pedal rendering behavior.

This change can make the score and playback better reflect sustained sound. It does not force longer notes when the recording has no sustain, and it does not guarantee accurate duration predictions before training.

## Parameter budget and defaults

| Component | Trainable parameters |
| --- | ---: |
| Short STFT CNN encoder | 706,944 |
| Raw FAN, normalization and temporal CNN | 2,803,776 |
| Fusion projection and normalization | 296,064 |
| Main four-layer bidirectional GRU | 3,537,408 |
| Main sequence normalization | 896 |
| One residual CNN block after the GRU | 2,412,480 |
| Pedal GRU, normalization and three heads | 173,443 |
| Four note heads | 165,792 |
| Offset refinement | 160,024 |
| Three global learned thresholds | 3 |
| **Total** | **10,256,830** |

STFT and Hann windows have no trainable parameters. Defaults are feature width 384, hidden size 224 per main direction, four main GRU layers, one residual block, pedal hidden size 64 per direction, dropout 0.2 and release persistence three frames.

`--short-fft` defaults to 2048 and accepts powers of two at least 512. `--fan-window` defaults to 8192 and accepts powers of two at least 2048; `--long-fft` is rejected for V11. Feature width must be at least 16 and divisible by eight. Main GRU hidden size must be at least four and divisible by four. Existing GRU layer/size and recurrence-backend options remain available. Changing dimensions changes the parameter count. Temporal Fourier modes/layers remain specific to V8.1.

## Training, validation and compatibility

V11 uses the existing supervised frame/onset/offset/release/velocity losses and pedal supervision. Pedal activity/onset/offset BCE contributions are grouped in the reported pedal component. The training total continues to equal the weighted recognizer components plus the weighted threshold contribution. Comparable unweighted frame BCE remains available for training and validation.

The three global frame/onset/offset thresholds are gradient-trained using **mean per-pitch F1, F2, F3, F4 and F5**. Their classifier probabilities are detached for threshold updates, preserving supervised recognition learning. Validation **AvgF loss** uses the same F1–F5 family and accumulates counts across all validation frames before scoring. It includes the existing penalty for activity on unsupported pitches and threshold-prior regularization. Minimum validation AvgF loss controls the best checkpoint and learning-rate scheduling; it is not a decoded note-duration score. Higher beta values emphasize recall, so false positives and fragmentation should also be inspected.

Equal pitch weights, ordinary random training windows, deterministic validation windows and disabled early stopping remain the defaults. Requested epochs run to completion unless interrupted or an error occurs. Previous architectures retain their own F-score definitions and key-release targets when resumed.

Fresh V11 training starts with **random weights** and writes `piano-v11.pt`, `piano-v11.last.pt` and `piano-v11.history.jsonl`. A compatible V11 checkpoint can initialize a new run or resume its optimizer, scheduler, epoch and random state. Older architectures cannot initialize V11: both network structure and target semantics differ. A previously trained V10.1 checkpoint does not become V11 merely by renaming its file.

The desktop app, CLI and local Python web app can load native V11 checkpoints. The browser-only Site/exporter retains its existing architecture limits. All earlier training commands remain in the README table.

## Verification and recommended next experiments

Five architecture tests passed on **2026-10-14**, covering the exact default parameter count, every branch/head's finite nonzero gradients, four-second/short/odd frame alignment, FAN context and amplitude behavior, checkpoint/configuration roundtrips, legacy parameter budgets and a default-size **CUDA forward/backward/AdamW step**. Run them with `python -m unittest tests.test_v11_model -v`.

These checks establish execution and architecture compatibility. They do not establish recognition quality. V11 still needs full training and evaluation against the same reference recordings as the other models.

Recommended future experiments, **not included in V11**, are soft regression onset/offset targets with timing interpolation, and decoded duration/fragmentation metrics for checkpoint monitoring. The current event targets remain the existing two-frame labels. Track note-offset accuracy, repeated-strike splitting, sustained durations, pedal intervals and per-pitch precision/recall when comparing V11 with ByteDance; changing F-score betas alone cannot resolve every duration error.
