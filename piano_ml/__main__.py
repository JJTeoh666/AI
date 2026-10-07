"""CLI: download, train, evaluate and transcribe piano audio."""

import argparse
import json
from pathlib import Path

from .download import download_additional, download_subset, download_validation
from .inference import transcribe_audio
from .training import train, score
from .evaluation import evaluate
from .piano_assets import install_piano
from .bass_data import download_bass
from .middle_data import refresh_middle_training


def threshold_list(value: str) -> list[float]:
    try:
        values = sorted(set(float(item) for item in value.split(",")))
        if not values or any(not 0 < item < 1 for item in values):
            raise ValueError
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated thresholds between 0 and 1") from error


def predict(args: argparse.Namespace) -> None:
    result = transcribe_audio(args.audio, args.checkpoint, args.threshold, args.device,
                              min_note_seconds=getattr(args, "min_note_seconds", None),
                              frame_threshold=getattr(args, "frame_threshold", None),
                              onset_threshold=getattr(args, "onset_threshold", None),
                              offset_threshold=getattr(args, "offset_threshold", None))
    content = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(content + "\n", encoding="utf-8")
        print(f"Wrote {args.output} ({len(result['notes'])} notes, {len(result['chords'])} chord spans)")
    else:
        print(content)


def download(args):
    replace_middle_gb = getattr(args, "replace_middle_gb", None)
    if replace_middle_gb is not None:
        if args.bass_gb is not None or args.additional_gb is not None or args.validation_total is not None or args.train_only:
            raise ValueError("Use --replace-middle-gb separately from other download modes.")
        return refresh_middle_training(args.data, replace_middle_gb, dry_run=args.dry_run)
    if args.train_only and (args.additional_gb is None or args.bass_gb is not None
                            or args.validation_total is not None):
        raise ValueError("Use --train-only with --additional-gb.")
    if args.bass_gb is not None:
        if args.additional_gb is not None or args.validation_total is not None:
            raise ValueError("Choose only one of --bass-gb, --additional-gb or --validation-total.")
        return download_bass(args.data, args.bass_gb, dry_run=args.dry_run)
    if args.validation_total is not None:
        if args.additional_gb is not None:
            raise ValueError("Choose --validation-total or --additional-gb.")
        return download_validation(args.data, args.validation_total, args.validation_budget_gb, args.seed)
    if args.additional_gb is not None:
        return download_additional(args.data, args.additional_gb, seed=args.seed,
                                   dry_run=args.dry_run, train_only=args.train_only)
    return download_subset(args.data, {"train": args.train, "validation": args.validation, "test": args.test})


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify active piano notes and recognizable chords")
    sub = parser.add_subparsers(dest="command", required=True)
    piano = sub.add_parser("download-piano", help="Install Salamander grand piano samples and a portable playback runtime")
    piano.set_defaults(func=lambda args: install_piano())
    dl = sub.add_parser("download", help="Fetch a small MAESTRO v3 subset")
    dl.add_argument("--data", default="data/maestro")
    dl.add_argument("--train", type=int, default=8)
    dl.add_argument("--validation", type=int, default=2)
    dl.add_argument("--test", type=int, default=2)
    dl.add_argument("--additional-gb", type=float, help="Add this many GB of usable files using a random, size-limited selection")
    dl.add_argument("--train-only", action="store_true", help="With --additional-gb, spend the entire budget on official training pairs")
    dl.add_argument("--bass-gb", type=float, help="Add this many GB of official training pairs ranked by rare bass-note coverage")
    dl.add_argument("--replace-middle-gb", type=float, help="Replace this many GB of active training recordings with new pairs rich in MIDI 30-68; keep old files as inactive backups")
    dl.add_argument("--dry-run", action="store_true", help="With budgeted download modes, save a plan without downloading audio files")
    dl.add_argument("--validation-total", type=int, help="Expand only the official validation split to this many local recordings")
    dl.add_argument("--validation-budget-gb", type=float, default=1.0, help="Additional validation download limit; default 1 GB")
    dl.add_argument("--seed", type=int, default=42)
    dl.set_defaults(func=download)
    tr = sub.add_parser("train", help="Train a PyTorch piano note model")
    tr.add_argument("--data", default="data/maestro")
    tr.add_argument("--output", help="Defaults to piano-v8.1.pt for the latest model; older architectures retain their versioned names")
    tr.add_argument("--architecture", choices=("auto", "frame", "onsets", "onsets-calibrated", "onsets-recurrent", "onsets-recurrent-global", "onsets-multires-global", "onsets-multires-balanced", "onsets-fourier-recurrent"), default="auto",
                    help="Auto resumes the saved architecture, or creates version 8.1 for a fresh run; V8 resumes continue as V8.1")
    tr.add_argument("--init-from", help="Transfer the compatible recognizer (or CNN features); optimizer and epoch start fresh")
    tr.add_argument("--hidden-size", type=int, help="Fresh GRU width: source width with --init-from; v8 default 192, older models 128")
    tr.add_argument("--gru-layers", type=int, help="Fresh GRU layers: source layers with --init-from, otherwise 2")
    tr.add_argument("--conv-steps", type=int, help="Shared convolution refinement steps; default 3 for recurrent architectures")
    tr.add_argument("--feature-width", type=int, help="V8 fused/Fourier channel width, default 384")
    tr.add_argument("--fourier-modes", type=int, help="V8 retained temporal Fourier modes including DC, default 9")
    tr.add_argument("--fourier-layers", type=int, help="V8 trainable Fourier mixing blocks, default 2")
    tr.add_argument("--long-fft", type=int, help="Version 6–8 long FFT size, default 8192")
    tr.add_argument("--release-frames", type=int, help="Version 6–8 consecutive inactive frames needed to end a note, default 3")
    tr.add_argument("--bass-sampling", type=float, help="Optional fraction of windows anchored on bass notes; default 0 for all versions")
    tr.add_argument("--bass-max-note", type=int, help="Highest MIDI pitch for bass sampling, default 47 (B2)")
    tr.add_argument("--treble-sampling", type=float, help="Optional fraction anchored on treble notes; default 0; bass + middle + treble must be <= 1")
    tr.add_argument("--middle-sampling", type=float, help="Optional fraction anchored in the middle focus range; default 0 for all versions")
    tr.add_argument("--middle-min-note", type=int, help="Inclusive lower MIDI pitch for middle focus (default 30)")
    tr.add_argument("--middle-max-note", type=int, help="Inclusive upper MIDI pitch for middle focus (default 68)")
    tr.add_argument("--treble-min-note", type=int, help="Lowest MIDI pitch for treble sampling, default 84 (C6)")
    tr.add_argument("--edge-loss-weight", type=float, help="Optional error multiplier on bass/treble keys; default 1 for equal pitch weighting")
    tr.add_argument("--middle-loss-weight", type=float, help="Optional error multiplier on middle keys; default 1 for equal pitch weighting")
    tr.add_argument("--offset-loss-weight", type=float, help="Offset loss scale: v6–v8 default 1; older models 0.5")
    tr.add_argument("--release-loss-weight", type=float, help="Additional frame loss around releases: v6–v8 default 0.25; older models 0")
    tr.add_argument("--threshold-calibration", choices=("training", "validation"),
                    help="V7/V8 use gradient training calibration; v6 defaults to decoded validation calibration; older models use smooth training calibration")
    tr.add_argument("--calibration-values", type=threshold_list,
                    help="Global validation candidates for each head, default 0.35,0.5,0.65,0.8; also includes current parameters")
    tr.add_argument("--calibration-every", type=int,
                    help="Full validation threshold search every N epochs, default 5; 1 searches every epoch. Current thresholds are evaluated every epoch")
    threshold_mode = tr.add_mutually_exclusive_group()
    threshold_mode.add_argument("--thresholds-only", dest="thresholds_only", action="store_true",
                               help="Train learned thresholds while freezing the pretrained recognizer")
    threshold_mode.add_argument("--joint-training", dest="thresholds_only", action="store_false",
                               help="Train the recognizer and learned thresholds together")
    tr.set_defaults(thresholds_only=None)
    threshold_learning = tr.add_mutually_exclusive_group()
    threshold_learning.add_argument("--freeze-thresholds", dest="freeze_thresholds", action="store_true",
                                   help="Hold learned threshold values fixed while training the recognizer")
    threshold_learning.add_argument("--learn-thresholds", dest="freeze_thresholds", action="store_false",
                                   help="Learn threshold parameters alongside the recognizer")
    tr.set_defaults(freeze_thresholds=None)
    tr.add_argument("--threshold-lr", type=float, help="Threshold parameter LR; fresh v7/v8 default 0.003, older models 0.01")
    tr.add_argument("--threshold-loss-weight", type=float, help="Smooth threshold loss multiplier (v8 pitch F0–F3, v7 F0/F3, older models F1); default 1")
    tr.add_argument("--threshold-temperature", type=float, help="Smooth decision temperature; default 0.1")
    tr.add_argument("--threshold-regularization", type=float, help="Pull thresholds toward initial values; default 0.05")
    tr.add_argument("--rnn-backend", choices=("auto", "native", "cudnn"), default="auto",
                    help="Auto uses native GPU recurrence on Windows nightly PyTorch builds")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--resume", help="Checkpoint to continue; --epochs is the number of additional epochs")
    tr.add_argument("--reset-optimizer", action="store_true", help="Resume weights with a fresh optimizer")
    tr.add_argument("--reset-early-stopping", action="store_true", help="Legacy option: reset non-improvement and scheduler counts while preserving the best-score target")
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--seconds", type=float, default=4.0)
    tr.add_argument("--windows-per-file", type=int, default=32)
    tr.add_argument("--max-files", type=int)
    tr.add_argument("--positive-weight", type=float, help="Frame loss positive weight; default 5 for onsets, 20 for frame")
    tr.add_argument("--event-weight", type=float, help="Onset/offset positive weight; defaults to 10")
    tr.add_argument("--lr", type=float, help="Default 0.0003 for onsets; resume preserves saved LR unless specified")
    tr.add_argument("--patience", type=int, default=0, help="Legacy compatibility option; ignored. Training runs all requested epochs")
    tr.add_argument("--lr-patience", type=int, default=3)
    tr.add_argument("--selection-metric", choices=("auto", "avgf_loss", "f1", "onset_f1", "note_f1", "note_f_avg", "note_macro_f03", "note_macro_f0123"), default="auto",
                    help="Auto selects minimum validation AvgF loss for V8.1, macro F0/F3 for V7, mean note F0..F4 for older onset models, or frame F1")
    tr.add_argument("--thresholds", type=threshold_list, help="Legacy shared frame/onset sweep")
    for head in ("frame", "onset", "offset"):
        tr.add_argument(f"--{head}-thresholds", type=threshold_list,
                        help=f"Independent {head} threshold candidates, comma-separated")
    tr.add_argument("--min-note-seconds", type=float)
    augmentation = tr.add_mutually_exclusive_group()
    augmentation.add_argument("--augment", dest="augment", action="store_true")
    augmentation.add_argument("--no-augment", dest="augment", action="store_false")
    tr.set_defaults(augment=None)
    tr.add_argument("--seed", type=int, help="Fresh runs choose a random seed; supply an integer to repeat initialization")
    tr.add_argument("--workers", type=int, default=0)
    tr.add_argument("--device", default="auto")
    tr.set_defaults(func=train)
    ev = sub.add_parser("evaluate", help="Frame and note metrics; calibrate thresholds on validation data")
    ev.add_argument("--data", default="data/maestro")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--split", choices=("validation", "test"), default="test")
    ev.add_argument("--max-files", type=int)
    ev.add_argument("--windows-per-file", type=int, default=16)
    ev.add_argument("--batch-size", type=int, default=4)
    ev.add_argument("--workers", type=int, default=0)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--seconds", type=float, default=4.0)
    ev.add_argument("--threshold", type=float, help="Legacy shared frame/onset override")
    ev.add_argument("--thresholds", type=threshold_list, help="Legacy shared frame/onset sweep; validation only")
    for head in ("frame", "onset", "offset"):
        ev.add_argument(f"--{head}-threshold", type=float, help=f"Override just the {head} threshold")
        ev.add_argument(f"--{head}-thresholds", type=threshold_list,
                        help=f"Sweep independent {head} values; validation only")
    ev.add_argument("--selection-metric", choices=("auto", "avgf_loss", "f1", "onset_f1", "note_f1", "note_f_avg", "note_macro_f03", "note_macro_f0123"), default="auto",
                    help="Auto selects minimum validation AvgF loss for V8.1, macro F0123 for legacy V8, macro F0/F3 for V7, or older models' note/frame scores")
    ev.add_argument("--min-note-seconds", type=float)
    ev.add_argument("--full-recordings", action="store_true", help="Evaluate complete recordings without clip boundaries")
    ev.add_argument("--calibrate-output", help="Save a checkpoint with the chosen validation threshold")
    ev.add_argument("--output", help="Save evaluation report JSON")
    ev.add_argument("--pitch-output", help="Save per-pitch metrics as CSV; --output also creates a .pitches.csv report automatically")
    ev.set_defaults(func=evaluate)
    pr = sub.add_parser("predict", help="Output timed note events and chord spans as JSON")
    pr.add_argument("audio", help="16-bit PCM WAV recording of a solo piano")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--output")
    pr.add_argument("--threshold", type=float, help="Legacy shortcut overriding both frame and onset")
    for head in ("frame", "onset", "offset"):
        pr.add_argument(f"--{head}-threshold", type=float, help=f"Override just the {head} threshold; default from checkpoint")
    pr.add_argument("--min-note-seconds", type=float)
    pr.add_argument("--device", default="auto")
    pr.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
