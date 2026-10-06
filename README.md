# Piano note and chord recognition with PyTorch

Identify piano notes and chords from WAV recordings, display the transcription on a piano grand staff, and play it with sampled piano sound.

The current model is **V8** (`onsets-fourier-recurrent`): a PyTorch network combining Fourier analysis, convolution and recurrence, with **10,012,924 trainable parameters**. It predicts 88 piano keys, note starts and releases, velocity, and sustain pedal.

## Contents

- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Desktop app and playback](#desktop-app-and-playback)
- [Command-line transcription](#command-line-transcription)
- [Evaluation](#evaluation)
- [V8 model and checkpoint selection](#v8-model-and-checkpoint-selection)
- [Troubleshooting](#troubleshooting)
- [Older models](#older-models)
- [Project files and checks](#project-files-and-checks)

## Setup

Use Python 3.10 or newer with a compatible PyTorch/TorchAudio pair. Follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your CPU or CUDA setup, then install the project requirements:

```powershell
cd C:\jj\AI
python -m pip install -r requirements.txt
```

Commands below run from the project directory. `--device auto` uses CUDA when available and otherwise uses the CPU. Tkinter is included with the standard Windows Python installer; Linux installations may need their distribution's Tkinter package.

## Data

Training uses [MAESTRO v3.0.0](https://magenta.withgoogle.com/datasets/maestro): 1,276 solo-piano recordings with aligned WAV audio and MIDI, about 199 hours in total. Official train, validation and test splits separate compositions. The dataset uses [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Current local inventory

Verified on **2026-10-06**:

| Split | Complete recordings |
| --- | ---: |
| Training | 53 |
| Validation | 16 |
| Test | 4 |
| **Total** | **73** |

Paired files occupy **3.885 GB**. The latest addition supplied **991 MB**, 12 training recordings, 93.6 minutes of audio and 55,649 piano note events, including 7,215 bass strikes and 4,471 treble strikes. All 24 added files passed size and CRC32 verification and were read through the training dataset. Details: [download verification](diagnostics/training-data-addition-2026-10-06.json).

Training includes every complete local pair in the official training split when `--max-files` is omitted. A running process keeps the file list it loaded at startup; restart or resume it to include newly downloaded recordings.

### Download data

On a new installation, fetch a small starting subset:

```powershell
python -m piano_ml download --data data/maestro --train 8 --validation 2 --test 2
```

Add about **1 GB of training data**:

```powershell
python -m piano_ml download --data data/maestro --additional-gb 1 --train-only
```

This skips existing complete files and adds only official training pairs. The budget measures extracted audio/MIDI bytes; compressed transfer is smaller, and the final size may be slightly below the budget. Add `--dry-run` to inspect the plan first. Running the command again adds another batch. Its manifest is `data/maestro/additional-download-plan.json`.

For recordings ranked by rare bass-note coverage:

```powershell
python -m piano_ml download --data data/maestro --bass-gb 1
```

This also uses only training pairs. It caches a separate, verified 56 MB MIDI index in `data/maestro/.bass-index/` and saves `bass-download-plan.json`. `--dry-run` prepares the index and plan without downloading audio.

To expand validation to a **target total** of 24 recordings:

```powershell
python -m piano_ml download --data data/maestro --validation-total 24 --validation-budget-gb 1 --seed 42
```

Changing validation recordings starts a new best-score comparison on resume. Without `--train-only`, `--additional-gb` allocates approximately 80% of added bytes to training, 10% to validation and 10% to test.

Downloads use HTTP range requests to fetch selected entries from official ZIP archives. To use the full dataset instead, extract the official archive so `data/maestro/maestro-v3.0.0.csv` and its year directories sit alongside each other. The full ZIP is about 101 GB compressed.

## Training

### Start a new run

```powershell
python -m piano_ml train --data data/maestro `
  --architecture onsets-fourier-recurrent --output checkpoints/piano-v8.pt `
  --epochs 100 --lr 0.0003 --threshold-lr 0.003 `
  --selection-metric note_macro_f0123 --lr-patience 6 `
  --windows-per-file 32 --batch-size 4 --augment --device auto
```

This starts V8 with random weights and three random global thresholds in 0.35–0.65. A new random seed is printed and saved; add `--seed 42` for repeatable initialization. Choose a new `--output` filename for an independent experiment.

With the current inventory and these settings, an epoch contains **1,696 training windows / 424 batches** and **256 fixed validation windows**. Windows are four seconds long. Training samples 30% of windows around bass notes, 30% around treble notes and 40% ordinarily, with augmentation. Validation uses ordinary deterministic windows.

### Resume training

Use the checkpoint from the **latest completed epoch**:

```powershell
python -m piano_ml train --data data/maestro `
  --resume checkpoints/piano-v8.last.pt --output checkpoints/piano-v8.pt `
  --epochs 50 --windows-per-file 32 --batch-size 4 --device auto
```

On resume, `--epochs 50` means **50 additional epochs**. Saved weights, thresholds, optimizer, scheduler, epoch, random state and training emphasis are restored. Keep the same `--seconds` and `--windows-per-file` to preserve the validation comparison. Auto resume retains the checkpoint's architecture, including older models.

Ctrl+C discards an incomplete epoch; the latest completed checkpoint remains available.

**After adding data or resuming a previously stopped run:** use the resume command above. Early stopping has been removed for all model versions; old checkpoint stop counts cannot end a new run. Existing commands containing `--patience` are still accepted, but that option is ignored.

Supply `--lr` or `--threshold-lr` to override a restored learning rate. `--reset-optimizer` deliberately starts fresh optimizer and scheduler states; use it when that is the intended experiment.

### Saved files and training length

| File | Use |
| --- | --- |
| `checkpoints/piano-v8.pt` | Best validation score; select this model in the app. |
| `checkpoints/piano-v8.last.pt` | Latest completed epoch; resume training from this file. |
| `checkpoints/piano-v8.history.jsonl` | Per-epoch scores, thresholds, learning rates and timing. |

For V8, **`note_macro_f0123`** controls best-model selection and learning rate scheduling. The best checkpoint changes when the score improves by more than `1e-6`. Training runs all requested epochs even when validation stops improving. The `stale_epochs` count remains in logs as a diagnostic. Learning rate reduction still uses `--lr-patience`.

Changing the validation files, window settings or selection metric starts a new comparison. Adding training recordings alone preserves comparison with the previous best.

### Comparing training and validation loss

Compare **`train_frame_bce` with `val_frame_bce`**. Both use unweighted binary cross-entropy over frame/key elements, with no positive-note or pitch-range weights. Each epoch averages over all elements, so a smaller final batch contributes proportionally.

Example output (illustrative numbers):

```text
epoch=101 train_total_loss=0.4875 train_frame_bce=0.0912 val_frame_bce=0.0798 ...
  train_loss_components: frame=0.1800 onset=0.0550 offset=0.0400 release=0.0110 velocity=0.0030 pedal=0.0017 threshold=0.1968
```

| Field | Meaning |
| --- | --- |
| `train_total_loss` | Console name for the complete optimization objective; saved as the existing `train_loss` history field. |
| `train_frame_bce` | Unweighted frame loss on the training predictions already computed during optimization. |
| `val_frame_bce` | Unweighted frame loss on validation; saved as `validation.frame_bce`. |
| `train_loss_components` | Seven contributions after applying their weights, averaged per batch. Their sum equals `train_loss`, within floating-point rounding. |
| `threshold_loss` | Existing raw threshold objective before applying `threshold_loss_weight`; the component report includes that multiplier. It is zero when threshold learning is disabled. |
| `validation.loss` | Existing unweighted frame loss averaged equally over batches, retained for compatibility. It can differ from `frame_bce` when batch sizes differ. |

Completed-epoch best/latest checkpoints also save `training_metrics` with `total_loss`, `frame_bce` and `loss_components`. Frame-only models report zero for unsupported components. New fields appear after restarting/resuming with this trainer; existing history lines remain unchanged, and older checkpoints can still resume.

Training still uses augmentation, dropout and bass/treble sampling; validation uses evaluation mode and deterministic ordinary windows. These conditions can create a gap even with the same frame-loss formula. Reporting reuses existing predictions and adds no model forward passes. Use validation **pitch F0123** to judge recognition quality and select the best V8 checkpoint.

## Desktop app and playback

```powershell
python app.py
```

1. Choose a checkpoint in **Model**. Use **Refresh** to discover saved models or **Browse model…** to open another checkpoint. The app prefers `piano-v8.pt` when it exists.
2. Select **Choose WAV…** and open an uncompressed 16-bit PCM piano recording.
3. Choose **Auto**, **CPU** or **CUDA**. Leave **Use model thresholds** checked, then select **Analyze**.
4. Open **Staff** for a grand staff with five treble lines and five bass lines. **Staff zoom** adjusts the notation size from 75% to 200% (starting at 125%). Scroll the main content with the mouse wheel or right-hand scrollbar to view the larger score. **Previous** and **Next** change pages; **Notation tempo (BPM)** and **Update staff** adjust the approximate notation.
5. Select the green **Play result** button in the fixed playback bar **at the bottom of the window**. Playback controls remain visible while the main content scrolls. Analyze a recording first, or load a saved transcription with **Open result JSON…**.

| Control or view | Purpose |
| --- | --- |
| **Play result** | Hear detected notes, including chords, predicted velocity and sustain pedal. |
| **Play original** | Hear the input WAV. |
| **Stop** | Stop playback or cancel result-audio preparation. |
| **Volume** | Adjust result and original playback from 0% (mute) to 100%, including while audio is playing. |
| **Gain** | Boost quiet result or original playback by 0 to +24 dB, including while audio is playing. |
| **Timeline** | Inspect the waveform, piano roll and chord spans. |
| **Notes / Chords** | Inspect detected events and timestamps. |
| **Export result WAV…** | Save the transcription as stereo, 44.1 kHz, 16-bit PCM audio. |
| **Export JSON…** | Save detected events. |
| **Save view…** | Export the current Staff or Timeline view as a PNG. |

Staff notation approximates rhythm in 4/4; playback follows the detected event times. The playback cursor follows the sound and advances staff pages automatically. Result playback works without the original recording.

The **Volume** slider sits beside **Stop** and shows the current percentage. It starts at 50% and can be increased to 100% or lowered to 0% (mute). It controls app playback. Exported WAV files retain their rendered volume.

The **Gain** slider below Volume starts at **0 dB**. Try **+6 dB** for about twice the signal amplitude, or **+12 dB** for about four times. Volume and gain apply together; 0% volume still mutes at any gain. Loud peaks are limited to the 16-bit range; reduce gain if they sound distorted. Gain affects playback, and exported WAVs retain their rendered volume.

To compare models, select another checkpoint and analyze the same audio again. Uncheck **Use model thresholds** to edit **Frame**, **Onset** and **Offset** individually; analyze again to apply the changes.

### Sampled piano sound

This workspace has **Salamander Grand Piano V3+20200602** and a portable **FluidSynth 2.6.1** Windows runtime installed. Result playback and WAV export use the same sampled Yamaha C5, with stereo sound, velocity layers, sustain, light reverb and a 2.5-second release tail.

To restore the assets or install them on another machine:

```powershell
python -m piano_ml download-piano
```

The piano download is approximately 310 MB and expands to a 1.27 GB SoundFont; the Windows runtime download is about 2.7 MB. Pinned SHA-256 checksums verify downloads, and playback works offline afterward. On Linux/macOS, install `libfluidsynth` through the system package manager.

The sound bank by Alexander Holm, converted for FreePats by Roberto, uses **CC BY 3.0**. Sources: [FreePats piano samples](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) and [FluidSynth runtime](https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.1). Local attribution and licenses are in `assets/piano/ATTRIBUTION.txt` and `assets/fluidsynth/LICENSE.txt`; provenance is in `assets/piano-playback.json`. If assets are missing, the app labels its fallback **Basic piano tone**.

## Command-line transcription

```powershell
python -m piano_ml predict path\to\solo-piano.wav `
  --checkpoint checkpoints/piano-v8.pt --output prediction.json --device auto
```

Input must be uncompressed **16-bit PCM WAV**, mono or stereo. The reader resamples it to 16 kHz mono. Output includes note names, MIDI pitches, start/end times, confidence, velocity, pedal events and recognized chord spans. Open the JSON in the app to view and play it.

## Evaluation

Evaluate complete validation recordings, then use the test split for the final assessment:

```powershell
python -m piano_ml evaluate --data data/maestro `
  --checkpoint checkpoints/piano-v8.pt --split validation --full-recordings `
  --output validation-v8.json --device auto

python -m piano_ml evaluate --data data/maestro `
  --checkpoint checkpoints/piano-v8.pt --split test --full-recordings `
  --output test-v8.json --device auto
```

Each `--output` also creates a `.pitches.csv` report, such as `validation-v8.pitches.csv`, with counts and scores for all 88 keys. Inspect `reference_count`, precision and recall to judge rare bass/treble pitches.

Reports include frame precision/recall/F1, onset F1, note F0–F4, macro per-key scores and the selection metric. A note match requires exact MIDI pitch, onset within 50 ms, and offset within `max(50 ms, 20% of reference duration)`.

Training evaluates fixed windows; `--full-recordings` avoids artificial clip boundaries and is useful for final comparisons. Compare models using the same recordings, metric and evaluation mode. For a V7/V8 comparison, explicitly select `--selection-metric note_macro_f0123`.

Threshold sweeps and `--calibrate-output` are restricted to validation data. See the [older-model calibration example](docs/model-versions.md#validation-threshold-calibration). Plain evaluation does not modify a checkpoint.

### Limits

MIDI labels represent keys held down; pedal is predicted separately. Chord names are derived from detected simultaneous notes using common-chord matches. Evaluation currently measures notes and frames; it does not establish velocity or pedal accuracy.

The model analyzes files offline using bidirectional context. The local dataset remains a small subset of MAESTRO, and recordings with other instruments, noise or unfamiliar pianos may perform worse. Trial checkpoints verify execution; their short runs do not establish recognition quality.

## V8 model and checkpoint selection

V8 combines two audio resolutions (2048- and 8192-sample FFTs), independent CNN encoders, Fourier Analysis Network layers, two learned temporal Fourier blocks and a two-layer bidirectional GRU. It produces frame, onset, offset, velocity and pedal predictions every **20 ms**.

Three global thresholds—frame, onset and offset—are learned by gradient descent on training targets and shared across all 88 keys. Validation evaluates the current values once per epoch.

For each key with reference notes, validation computes:

```text
F_beta = (1 + beta²) * TP / ((1 + beta²) * TP + FP + beta² * FN)
pitch F0123 = (F0 + F1 + F2 + F3) / 4
note_macro_f0123 = mean(pitch F0123 across keys with reference notes)
```

F0 is precision, F1 balances precision and recall, and F2/F3 give recall more weight. These names describe F-beta metrics. Every supported key contributes equally; undefined scores are zero. False positives on keys without references remain in the micro metrics and per-key reports.

See [V8 structure, parameter budget and verification](docs/model-v8.md) for the full diagram, loss/threshold details and trial records.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing `piano-v8.last.pt` | Check `checkpoints/` for the actual filename. The latest checkpoint exists only after a completed epoch; use an existing compatible checkpoint or start a new run. |
| A previous run stopped early | Restart/resume with the updated trainer. Early stopping is removed; `--epochs` sets the training length. |
| New recordings are absent | Restart/resume training and omit `--max-files`. The startup message reports the recording count. |
| CUDA runs out of memory | Reduce `--batch-size` to 2 or 1. |
| Training takes longer after a download | More recordings add batches. Inspect `train_seconds` and `validation_seconds` in the history before changing settings. |
| Recurrent backend fails on Windows nightly PyTorch | `--rnn-backend auto` uses native GPU recurrence on those builds. Native recurrence may be slower. |
| Cannot play the result | Analyze audio or load result JSON first, then use **Play result** in the fixed bottom playback bar. Restore samples with `download-piano` if needed. |
| `Glyph ... missing from font(s) DejaVu Sans` | A chart filename or label uses a character outside the font. The app uses installed CJK font fallbacks; restart after updating. On systems without a CJK font, install one such as Noto Sans CJK. This warning affects text display. |
| WAV format error | Convert the input to uncompressed 16-bit PCM WAV. |

## Older models

Fresh `--architecture auto` creates V8. Resume restores the saved architecture. V5–V7 recognizers have different structures from V8; start V8 fresh or use `--init-from` with another compatible V8 checkpoint.

| Version | Architecture | Default selection metric |
| --- | --- | --- |
| V8 | `onsets-fourier-recurrent` | `note_macro_f0123` |
| [V7](docs/model-versions.md#version-7) | `onsets-multires-balanced` | `note_macro_f03` |
| [V6](docs/model-versions.md#version-6) | `onsets-multires-global` | `note_f_avg` |
| [V5](docs/model-versions.md#version-5) | `onsets-recurrent-global` | `note_f_avg` |
| [V4](docs/model-versions.md#version-4) | `onsets-recurrent` | `note_f_avg` |
| [V3](docs/model-versions.md#version-3) | `onsets-calibrated` | `note_f_avg` |
| [V2](docs/model-versions.md#version-2) | `onsets` | `note_f_avg` |
| [Frame baseline](docs/model-versions.md#frame-model) | `frame` | Frame `f1` |

[Older training commands and experiment history](docs/model-versions.md) include transfers, register thresholds, validation calibration and saved trial comparisons.

## Project files and checks

| Path | Contents |
| --- | --- |
| `piano_ml/` | Models, training, data loading, decoding, evaluation and app code. |
| `app.py` | Desktop app entry point. |
| `data/maestro/` | Official CSV metadata, paired recordings and download manifests. |
| `checkpoints/` | Best/latest models and training history. |
| `assets/` | Sampled piano sound and playback runtime. |
| `diagnostics/` | Data verification and model trial reports. |
| `docs/` | Detailed V8 and older-version references. |
| `tests/` | Automated checks. |

Run the checks:

```powershell
python -m unittest discover -s tests -v
```

Design references: [Onsets and Frames](https://arxiv.org/abs/1710.11153), [Piano Transcription with Pedals](https://arxiv.org/abs/2010.01815), [Fourier Analysis Networks](https://arxiv.org/abs/2410.02675) and [Fourier Neural Operators](https://arxiv.org/abs/2010.08895).
