# Version 11.1 model reference

[README](../README.md) · [Training command table](../README.md#training-commands) · [V11 reference](model-v11.md)

V11.1 is `onsets-dual-stft-sustain-recurrent`, with **9,929,470 trainable parameters** at default settings. Use its explicit architecture flag for a fresh V11.1 run; fresh auto training now selects V12.1. It replaces V11's raw FAN branch with an independent long STFT/CNN branch. The four-layer recurrent stack, residual CNN, seven output heads and sustain semantics remain the same. Model version is `11.1`; checkpoint format version is `14`.

## Structure

```text
16 kHz mono audio
  ├─ Short STFT 2048 → CNN → 384 features ──┐
  └─ Long STFT 8192 → CNN → 384 features ───┘
       → concatenate 768 → Linear 768→384 / LayerNorm / GELU / dropout
            ├─ Four-layer bidirectional GRU, 224 units per direction
            │    → LayerNorm → one residual CNN block, 448→896→448
            │    → frame / onset / offset / velocity
            └─ One-layer bidirectional pedal GRU, 64 per direction
                 → LayerNorm → pedal activity / pedal onset / pedal offset
```

The main residual CNN follows all four recurrent layers. The separate pedal branch reads the fused acoustic features. Frame, onset, offset and velocity each output **88 logits per frame**; each of the three pedal heads outputs **one**. Onset probabilities condition frame predictions, while frame/onset/pedal probabilities participate in offset refinement.

## Spectral context and alignment

| Branch | Hann window | Frequency bins | Frequency spacing | Positions after CNN pooling |
| --- | --- | ---: | ---: | ---: |
| Short STFT | 2048 samples / **128 ms** | 1025 | 7.8125 Hz | 17 |
| Long STFT | 8192 samples / **512 ms** | 4097 | 1.953125 Hz | 65 |

Both branches retain the complete, linearly spaced magnitude spectrum; neither uses mel filtering, CQT, harmonic stacking or a FAN. The long branch has the same default context duration as V11's raw FAN input, with fixed spectral features in its place.

Both STFTs use centered periodic Hann windows, constant zero padding and the same **320-sample / 20 ms hop**. Frame centers align at sample 0, 320, 640 and so on. Output length is `floor(samples / 320) + 1`: four seconds produces **201 frames for every head**. Both spectral calculations stay in float32 under autocast.

Each magnitude is divided by its fixed Hann-window sum (`n_fft / 2`) and transformed with `log1p(10 * magnitude)`. An exact-bin sinusoid has the same amplitude convention in both branches. No individual waveform window is normalized using its measured loudness.

Independent CNNs use channels 16, 32, 64 and 96, 3×3 convolutions, eight-group GroupNorm, GELU and frequency-only pooling factors 2, 2, 2 and 8. Ceiling pooling retains upper frequency bins. Each encoder projects its flattened frequency/channel features to 384 features per frame.

The default training window remains **four seconds**. Bidirectional recurrence uses earlier and later features within the analysis section, so this remains an offline transcription model.

## Parameter budget and options

| Component | Trainable parameters |
| --- | ---: |
| Short STFT CNN encoder | 706,944 |
| Long STFT CNN encoder | 2,476,416 |
| Fusion projection and normalization | 296,064 |
| Main four-layer bidirectional GRU | 3,537,408 |
| Main sequence normalization | 896 |
| Residual CNN after the GRU | 2,412,480 |
| Pedal GRU, normalization and three heads | 173,443 |
| Four note heads | 165,792 |
| Offset refinement | 160,024 |
| Three global thresholds | 3 |
| **Total** | **9,929,470** |

STFTs and Hann windows have no trainable parameters. Defaults remain feature width 384, hidden size 224 per main GRU direction, four main layers, one residual block, pedal hidden size 64 per direction, dropout 0.2 and release persistence three frames.

`--short-fft` defaults to 2048 and accepts powers of two at least 512. `--long-fft` defaults to 8192 and accepts powers of two at least 2048. `--fan-window` is specific to V11 and is rejected for V11.1. Existing feature width, GRU size/layer and recurrence-backend controls remain available. Changing dimensions changes the parameter budget.

## Sustain, training and compatibility

V11.1 retains V11's **`sustain_extended`** note targets. Key releases under sustain extend to pedal release, capped by the next strike of the same pitch. Transformation occurs on the complete performance before selecting windows. Pedal activity, pedal-down and pedal-up receive separate supervision. Already-extended result durations are not extended a second time during playback.

The supervised loss components and three gradient-trained global thresholds are unchanged. The smooth threshold objective and validation AvgF loss use **F1, F2, F3, F4 and F5** per pitch, including unsupported-pitch activity penalties and threshold-prior regularization. Minimum validation AvgF loss controls the best checkpoint. This is distinct from a decoded note-duration score.

Fresh V11.1 starts with random weights. New files are `piano-v11.1.pt`, `piano-v11.1.last.pt` and `piano-v11.1.history.jsonl`. Resume restores the checkpoint's architecture and saved training state. Only compatible V11.1 checkpoints can initialize a new V11.1 run; V11 has a different frontend and cannot transfer through `--init-from`. V11 itself retains **10,256,830 parameters** and its original state keys. Earlier version commands remain in the README table.

## Verification

Fresh V11.1 reduces learning rates after **15 consecutive non-improving epochs**, using absolute `1e-6` improvement tolerance to match best-model saving. Omitting `--lr-patience` on resume restores the saved scheduler rule; an explicit override applies after restoration without clearing the accumulated bad-epoch counter. Early stopping remains disabled. See the README for the V11 training audit and its timing/convergence limits.

Five architecture tests passed on **2026-10-15**: exact parameter count; independent frontend shapes; recurrent-before-residual ordering; 201-frame alignment; finite nonzero gradients for every parameter; spectral amplitude, frequency and context checks; short/odd inputs; checkpoint/configuration roundtrip; older parameter budgets; and a default-size CUDA forward/backward/AdamW step. Both FFTs retain float32 under CUDA autocast.

Run the model checks with `python -m unittest tests.test_v111_model -v`. These checks establish implementation behavior; full training and recognition-quality evaluation are still required. The frontend replacement does not by itself guarantee faster training, longer notes or higher accuracy.
