"""Training, validation threshold calibration, and resumable checkpoints."""

import hashlib
import json
import math
import random
import secrets
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import MaestroWindows, collate_windows
from .middle_data import FOCUS_LOW_NOTE, FOCUS_HIGH_NOTE
from .calibration import Calibration
from .avgf import avgf_accumulators
from .inference import get_device
from .thresholds import Thresholds, saved_thresholds, threshold_grid, requested_grid
from .model import (N_NOTES, LOW_NOTE, HIGH_NOTE, SAMPLE_RATE, HOP_LENGTH, build_model,
                    model_config, DEFAULT_ARCHITECTURE, FOURIER_ARCHITECTURE, FOURIER_MODEL_VERSION,
                    BALANCED_ARCHITECTURES, RECURRENT_CONV_ARCHITECTURES,
                    MULTIRES_ARCHITECTURES, default_selection_metric, checkpoint_format_version)

LOSS_COMPONENTS = ("frame", "onset", "offset", "release", "velocity", "pedal", "threshold")
PITCH_EMPHASIS_VERSION = 2


class _FrameBCEAccumulator:
    """Unweighted frame BCE, with element and legacy batch mean reductions."""

    def __init__(self):
        self.element_total = None
        self.batch_total = None
        self.elements = 0
        self.batches = 0

    @torch.no_grad()
    def add(self, logits, targets):
        value = nn.functional.binary_cross_entropy_with_logits(logits, targets).to(torch.float64)
        count = targets.numel()
        if self.element_total is None:
            self.element_total = value * count
            self.batch_total = value.clone()
        else:
            self.element_total += value * count
            self.batch_total += value
        self.elements += count
        self.batches += 1

    def mean(self):
        return float(self.element_total / self.elements)

    def batch_mean(self):
        return float(self.batch_total / self.batches)


def save_checkpoint(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state() -> dict:
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    if device.type == "cuda" and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def score(model: nn.Module, loader: DataLoader, device: torch.device,
          thresholds=(0.5,), selection_metric="f1", min_note_seconds=None,
          threshold_configs=None, avgf_temperature=0.1, avgf_regularization=0.05) -> dict:
    """Calibrate on fixed validation windows, aggregating micro and pitch metrics."""
    model.eval()
    calibration = Calibration(threshold_configs if threshold_configs is not None else
                              [model.learned_thresholds()] if hasattr(model, "learned_thresholds") else
                              threshold_grid(Thresholds(), shared_values=thresholds), min_note_seconds,
                              release_frames=model_config(model).get("release_frames", 2))
    avgf = avgf_accumulators(model, calibration.grid, avgf_temperature, avgf_regularization)
    if selection_metric == "avgf_loss" and not avgf:
        raise ValueError("AvgF loss selection requires a V8/V8.1 Fourier model.")
    frame_loss = _FrameBCEAccumulator()
    with torch.inference_mode():
        for waves, labels in loader:
            waves = waves.to(device)
            truth = labels["frame"] if isinstance(labels, dict) else labels
            output = model(waves)
            output = output if isinstance(output, dict) else {"frame": output}
            frame_loss.add(output["frame"], truth.to(device))
            probabilities = {key: logits.sigmoid().cpu().numpy() for key, logits in output.items()}
            if avgf:
                if not isinstance(labels, dict) or any(head not in labels for head in ("frame", "onset", "offset")):
                    raise ValueError("AvgF validation requires frame, onset and offset targets.")
                for accumulator in avgf.values():
                    accumulator.add(probabilities, {head: labels[head].numpy() for head in ("frame", "onset", "offset")})
            truth = truth.numpy() >= 0.5
            duration = waves.shape[1] / SAMPLE_RATE
            for index in range(len(waves)):
                reference = labels.get("reference_notes", [None] * len(waves))[index] if isinstance(labels, dict) else None
                calibration.add({key: value[index] for key, value in probabilities.items()},
                                truth[index], reference, duration, window=True)
    return {"loss": frame_loss.batch_mean(), "frame_bce": frame_loss.mean(),
            **calibration.results(selection_metric, {values: accumulator.results() for values, accumulator in avgf.items()})}


def pitch_loss_weights(bass_max_note=47, treble_min_note=84, edge_loss_weight=1.0,
                       device=None, middle_loss_weight=1.0,
                       middle_min_note=FOCUS_LOW_NOTE, middle_max_note=FOCUS_HIGH_NOTE):
    """Weight positive and negative errors; explicit middle emphasis overrides edges."""
    if not LOW_NOTE <= bass_max_note < treble_min_note <= HIGH_NOTE:
        raise ValueError("Focus ranges must be disjoint piano MIDI ranges (21–108).")
    if not math.isfinite(edge_loss_weight) or edge_loss_weight <= 0:
        raise ValueError("Edge loss weight must be finite and positive.")
    if not math.isfinite(middle_loss_weight) or middle_loss_weight <= 0:
        raise ValueError("Middle loss weight must be finite and positive.")
    if not LOW_NOTE <= middle_min_note <= middle_max_note <= HIGH_NOTE:
        raise ValueError("Middle focus must be an inclusive piano MIDI range (21-108).")
    pitches = torch.arange(LOW_NOTE, HIGH_NOTE + 1, device=device)
    weights = torch.where((pitches <= bass_max_note) | (pitches >= treble_min_note),
                          float(edge_loss_weight), 1.0)
    if middle_loss_weight != 1:
        weights = torch.where((pitches >= middle_min_note) & (pitches <= middle_max_note),
                              float(middle_loss_weight), weights)
    return weights


def training_loss_components(output, targets, positive_weight=5.0, event_weight=10.0,
                             offset_loss_weight=0.5, release_loss_weight=0.0, pitch_weights=None):
    """Return the six weighted contributions to the recognizer objective."""
    frame = output["frame"] if isinstance(output, dict) else output
    weight = frame.new_full((N_NOTES,), positive_weight)

    def pitch_bce(logits, truth, positive):
        if pitch_weights is None:
            return nn.functional.binary_cross_entropy_with_logits(logits, truth, pos_weight=positive)
        errors = nn.functional.binary_cross_entropy_with_logits(logits, truth, pos_weight=positive, reduction="none")
        return (errors * pitch_weights).sum() / (pitch_weights.sum() * errors.numel() / N_NOTES)

    components = {name: frame.new_zeros(()) for name in LOSS_COMPONENTS[:-1]}
    components["frame"] = pitch_bce(frame, targets["frame"], weight)
    if not isinstance(output, dict):
        return components
    event_positive = frame.new_full((N_NOTES,), event_weight)
    for key, scale in (("onset", 1.0), ("offset", offset_loss_weight)):
        components[key] = scale * pitch_bce(output[key], targets[key], event_positive)
    if release_loss_weight:
        # Supervise activity just before and after actual key releases. This
        # penalizes both premature endings and notes held past their reference.
        boundary = nn.functional.max_pool1d(targets["offset"].transpose(1, 2), 5, stride=1, padding=2).transpose(1, 2)
        if pitch_weights is not None:
            boundary = boundary * pitch_weights
        frame_errors = nn.functional.binary_cross_entropy_with_logits(
            frame, targets["frame"], pos_weight=weight, reduction="none")
        components["release"] = release_loss_weight * (frame_errors * boundary).sum() / boundary.sum().clamp_min(1)
    mask = targets["onset"]
    if pitch_weights is not None:
        mask = mask * pitch_weights
    velocity = ((output["velocity"].sigmoid() - targets["velocity"]) ** 2 * mask).sum() / mask.sum().clamp_min(1)
    components["velocity"] = 0.5 * velocity
    components["pedal"] = 0.2 * nn.functional.binary_cross_entropy_with_logits(
        output["pedal"], targets["pedal"], pos_weight=frame.new_tensor([2.0]))
    return components


def training_loss(output, targets, positive_weight=5.0, event_weight=10.0,
                  offset_loss_weight=0.5, release_loss_weight=0.0, pitch_weights=None):
    """Keep the scalar objective API used by existing training callers."""
    return sum(training_loss_components(output, targets, positive_weight, event_weight,
                                        offset_loss_weight, release_loss_weight, pitch_weights).values())


def transfer_weights(model, source):
    """Preserve the whole compatible recognizer when adding threshold parameters."""
    target = model.state_dict()
    optional = ("threshold_module.", "recurrent_blocks.", "long_", "offset_refinement.")
    recognizer_keys = [key for key in target if not key.startswith(optional)]
    complete = all(key in source["model"] and target[key].shape == source["model"][key].shape
                   for key in recognizer_keys)
    transferred = {key: value for key, value in source["model"].items()
                   if key in target and target[key].shape == value.shape
                   and (complete or key.startswith(("features.", "mel.")))}
    if not transferred:
        raise ValueError("The initialization checkpoint has no compatible CNN features.")
    model.load_state_dict(transferred, strict=False)
    if complete and hasattr(model, "long_scale") and "long_scale" not in transferred:
        with torch.no_grad():
            model.long_scale.zero_()
    if hasattr(model, "threshold_module") and "threshold_module.raw" not in transferred:
        model.threshold_module.initialize(saved_thresholds(source))
    return complete


def train(args) -> None:
    if args.epochs < 1 or args.positive_weight is not None and args.positive_weight <= 0:
        raise ValueError("Epoch count and positive weight must be positive.")
    resume = getattr(args, "resume", None)
    init_from = getattr(args, "init_from", None)
    if resume and init_from:
        raise ValueError("Choose either --resume or --init-from.")
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else {}
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = saved.get("training_config", {}).get("seed", 42) if resume else secrets.randbits(32)
    if not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be an integer between 0 and 4294967295.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    requested = getattr(args, "architecture", "frame")
    config = saved.get("model_config", {"architecture": "frame"}) if resume else {
        "architecture": DEFAULT_ARCHITECTURE if requested == "auto" else requested}
    if resume and requested not in ("auto", config["architecture"]):
        raise ValueError("Resume architecture differs from checkpoint. Use --init-from to transfer compatible weights.")
    initialization = torch.load(init_from, map_location="cpu", weights_only=True) if init_from else None
    if not resume and config["architecture"] != "frame":
        source_config = (initialization or {}).get("model_config", {})
        fourier = config["architecture"] == FOURIER_ARCHITECTURE
        if fourier and initialization and source_config.get("architecture") != FOURIER_ARCHITECTURE:
            raise ValueError("Version 8 has a new structure. Start fresh, or use --init-from with another version 8 checkpoint.")
        config.update(hidden_size=getattr(args, "hidden_size", None) or source_config.get("hidden_size", 192 if fourier else 128),
                      gru_layers=getattr(args, "gru_layers", None) or source_config.get("gru_layers", 2),
                      dropout=source_config.get("dropout", 0.2),
                      rnn_backend=getattr(args, "rnn_backend", "auto"))
        if config["architecture"] in RECURRENT_CONV_ARCHITECTURES:
            conv_steps = getattr(args, "conv_steps", None)
            config["conv_steps"] = source_config.get("conv_steps", 3) if conv_steps is None else conv_steps
        if config["architecture"] in MULTIRES_ARCHITECTURES:
            for name, default in (("long_fft", 8192), ("release_frames", 3)):
                value = getattr(args, name, None)
                config[name] = source_config.get(name, default) if value is None else value
        if fourier:
            for name, default in (("feature_width", 384), ("fourier_modes", 9), ("fourier_layers", 2)):
                value = getattr(args, name, None)
                config[name] = source_config.get(name, default) if value is None else value
    if not resume and getattr(args, "conv_steps", None) is not None and config["architecture"] not in RECURRENT_CONV_ARCHITECTURES:
        raise ValueError("--conv-steps requires --architecture onsets-recurrent.")
    if any(getattr(args, name, None) is not None for name in ("feature_width", "fourier_modes", "fourier_layers")) and config["architecture"] != FOURIER_ARCHITECTURE:
        raise ValueError("Fourier configuration requires --architecture onsets-fourier-recurrent.")
    device = get_device(args.device)
    model = build_model(config).to(device)
    if resume:
        model.load_state_dict(saved["model"])
    if init_from:
        complete_transfer = transfer_weights(model, initialization)
        print(f"Transferred {'complete recognizer' if complete_transfer else 'CNN features'} from {init_from}; "
              "optimizer and epoch start fresh", flush=True)
    architecture = config["architecture"]
    fourier = architecture == FOURIER_ARCHITECTURE
    balanced = architecture in BALANCED_ARCHITECTURES
    multires = architecture in MULTIRES_ARCHITECTURES
    learned = hasattr(model, "threshold_module")
    global_thresholds = learned and model.threshold_module.scope == "global"
    suffix = f"-v{FOURIER_MODEL_VERSION}" if fourier else "-v7" if balanced else "-v6" if multires else "-v5" if global_thresholds else "-v4" if architecture == "onsets-recurrent" else "-v3" if learned else "-v2" if architecture == "onsets" else ""
    output = Path(args.output or f"checkpoints/piano{suffix}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=True)
        if existing.get("model_config", {"architecture": "frame"}) != model_config(model):
            raise ValueError("Output contains a different architecture. Choose a new --output filename.")
    latest = output.with_name(output.stem + ".last" + output.suffix)
    augment = getattr(args, "augment", None)
    if augment is None:
        augment = saved.get("training_config", {}).get("augment", architecture != "frame")
    prior = saved.get("training_config", {})
    thresholds_only = getattr(args, "thresholds_only", None)
    thresholds_only = prior.get("thresholds_only", False) if thresholds_only is None else thresholds_only
    if thresholds_only:
        if not learned or not (resume or init_from and complete_transfer):
            raise ValueError("--thresholds-only requires a calibrated model and a complete trained recognizer via --resume or --init-from.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("threshold_module."))
        if getattr(args, "augment", None) is None:
            augment = prior.get("augment", False)
    freeze_thresholds = getattr(args, "freeze_thresholds", None)
    freeze_thresholds = prior.get("freeze_thresholds", False) if freeze_thresholds is None else freeze_thresholds
    if thresholds_only and getattr(args, "freeze_thresholds", None) is None:
        freeze_thresholds = False
    if freeze_thresholds:
        if not learned or thresholds_only:
            raise ValueError("--freeze-thresholds requires a learned-threshold model in joint training mode.")
        model.threshold_module.requires_grad_(False)
    calibration_mode = getattr(args, "threshold_calibration", None) or prior.get(
        "threshold_calibration", "validation" if multires and not balanced else "training")
    validation_calibration = calibration_mode == "validation"
    if balanced and validation_calibration:
        raise ValueError("Versions 7/8 use gradient threshold learning; use --threshold-calibration training.")
    if validation_calibration:
        if not global_thresholds or thresholds_only or freeze_thresholds:
            raise ValueError("Validation threshold calibration requires three global thresholds in joint training, without --freeze-thresholds.")
        model.threshold_module.requires_grad_(False)
    calibration_values = getattr(args, "calibration_values", None) or prior.get("calibration_values", [0.35, 0.5, 0.65, 0.8])
    if any(not math.isfinite(value) or not 0.05 < value < 0.95 for value in calibration_values):
        raise ValueError("Calibration values must be inside threshold parameter bounds (0.05, 0.95).")
    calibration_every = getattr(args, "calibration_every", None)
    calibration_every = prior.get("calibration_every", 5) if calibration_every is None else calibration_every
    if not isinstance(calibration_every, int) or calibration_every < 1:
        raise ValueError("Calibration interval must be a positive integer.")
    training_options = {}
    reset_pitch_emphasis = prior.get("pitch_emphasis_version", 0) < PITCH_EMPHASIS_VERSION
    pitch_emphasis_defaults = {"bass_sampling": 0.0, "treble_sampling": 0.0, "middle_sampling": 0.0,
                               "edge_loss_weight": 1.0, "middle_loss_weight": 1.0}
    for name, default in (("bass_sampling", 0.0), ("bass_max_note", 47),
                          ("treble_sampling", 0.0), ("treble_min_note", 84),
                          ("middle_sampling", 0.0),
                          ("middle_min_note", FOCUS_LOW_NOTE), ("middle_max_note", FOCUS_HIGH_NOTE),
                          ("edge_loss_weight", 1.0), ("middle_loss_weight", 1.0),
                          ("offset_loss_weight", 1.0 if multires else 0.5),
                          ("release_loss_weight", 0.25 if multires else 0.0)):
        value = getattr(args, name, None)
        # Pre-uniform checkpoints must not silently restore their old pitch emphasis.
        previous = {} if reset_pitch_emphasis and name in pitch_emphasis_defaults else prior
        training_options[name] = previous.get(name, default) if value is None else value
    if any(not math.isfinite(training_options[name]) or training_options[name] < 0
           for name in ("offset_loss_weight", "release_loss_weight")):
        raise ValueError("Offset and release loss weights must be finite and nonnegative.")
    pitch_weights = None
    if training_options["edge_loss_weight"] != 1 or training_options["middle_loss_weight"] != 1:
        pitch_weights = pitch_loss_weights(training_options["bass_max_note"],
            training_options["treble_min_note"], training_options["edge_loss_weight"], device,
            middle_loss_weight=training_options["middle_loss_weight"],
            middle_min_note=training_options["middle_min_note"], middle_max_note=training_options["middle_max_note"])
    threshold_options = {name: getattr(args, name, None) if getattr(args, name, None) is not None
                         else prior.get(name, default) for name, default in (
                             ("threshold_lr", 0.003 if balanced else 0.01), ("threshold_loss_weight", 1.0),
                             ("threshold_temperature", 0.1), ("threshold_regularization", 0.05))}
    if any(not math.isfinite(value) or value <= 0 for name, value in threshold_options.items() if name != "threshold_regularization") or not math.isfinite(threshold_options["threshold_regularization"]) or threshold_options["threshold_regularization"] < 0:
        raise ValueError("Threshold learning settings must be finite and positive; regularization can be zero.")
    positive_weight = args.positive_weight if args.positive_weight is not None else prior.get(
        "positive_weight", 5.0 if architecture != "frame" else 20.0)
    event_weight = getattr(args, "event_weight", None)
    event_weight = prior.get("event_weight", 10.0) if event_weight is None else event_weight
    train_set = MaestroWindows(args.data, "train", args.seconds, args.windows_per_file,
                               random_windows=True, max_files=args.max_files,
                               multi_target=True, augment=augment,
                               bass_sampling=training_options["bass_sampling"], bass_max_note=training_options["bass_max_note"],
                               treble_sampling=training_options["treble_sampling"], treble_min_note=training_options["treble_min_note"],
                               middle_sampling=training_options["middle_sampling"],
                               middle_min_note=training_options["middle_min_note"], middle_max_note=training_options["middle_max_note"])
    val_set = MaestroWindows(args.data, "validation", args.seconds, max(1, args.windows_per_file // 2),
                             max_files=args.max_files, multi_target=True)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers, "collate_fn": collate_windows}
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, **loader_options)
    metric = getattr(args, "selection_metric", "auto")
    if metric == "auto":
        metric = default_selection_metric(architecture)
    selection_mode = "min" if metric == "avgf_loss" else "max"
    if metric == "avgf_loss" and not fourier:
        raise ValueError("AvgF loss selection requires a V8/V8.1 Fourier model.")
    if learned and any(getattr(args, name, None) is not None for name in (
            "thresholds", "frame_thresholds", "onset_thresholds", "offset_thresholds")):
        raise ValueError("The calibrated architecture learns its thresholds. Omit threshold grids while training.")
    grid = None if learned else requested_grid(args, saved_thresholds(saved), architecture, training=True, prior=prior)
    minimum = getattr(args, "min_note_seconds", None)
    if minimum is None:
        minimum = saved.get("min_note_seconds", 0.04 if architecture != "frame" else 0.08)
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("Minimum note duration must be nonnegative and finite.")
    validation_settings = {"files": [row["audio_filename"] for row in val_set.records],
        "seconds": args.seconds, "windows": val_set.windows_per_file, "metric": metric,
        "threshold_grid": "learned_global" if global_thresholds else "learned_registers" if learned else [values.to_dict() for values in grid],
        "minimum": minimum}
    if multires or validation_calibration:
        validation_settings.update(release_frames=config.get("release_frames", 2),
                                   calibration_values=calibration_values if validation_calibration else [])
    if metric == "avgf_loss":
        validation_settings["avgf"] = {"temperature": threshold_options["threshold_temperature"],
                                       "regularization": threshold_options["threshold_regularization"],
                                       "prior": model.threshold_module.prior.detach().cpu().tolist(),
                                       "aggregation": "all_validation_frames", "pitch_weights": "equal"}
    signature = hashlib.sha256(json.dumps(validation_settings, sort_keys=True).encode()).hexdigest()
    comparable = saved.get("validation_signature") == signature and output.exists()
    if resume and saved.get("selection_metric") != metric:
        print(f"Selection metric changed to {metric}; resetting best-score comparison and non-improvement count", flush=True)
    lr = args.lr if args.lr is not None else (3e-4 if architecture != "frame" else 1e-3)
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(positive_weight) or positive_weight <= 0 or not math.isfinite(event_weight) or event_weight <= 0:
        raise ValueError("Learning rate and loss positive weights must be positive and finite.")
    groups = [{"params": [parameter for name, parameter in model.named_parameters()
                          if parameter.requires_grad and not name.startswith("threshold_module.")],
               "lr": lr, "weight_decay": 1e-4, "name": "recognizer"}]
    if learned:
        groups.append({"params": [parameter for parameter in model.threshold_module.parameters() if parameter.requires_grad],
                       "lr": threshold_options["threshold_lr"],
                       "weight_decay": 0.0, "name": "thresholds"})
    optimizer = torch.optim.AdamW(groups, lr=lr)
    mode_changed = bool(saved) and (prior.get("thresholds_only", False) != thresholds_only
                                   or prior.get("freeze_thresholds", False) != freeze_thresholds
                                   or prior.get("threshold_calibration", "training") != calibration_mode)
    restoring_optimizer = "optimizer" in saved and not getattr(args, "reset_optimizer", False) and not mode_changed
    if mode_changed:
        comparable = False
        print("Training mode changed; starting fresh optimizer and validation comparison", flush=True)
    if restoring_optimizer:
        optimizer.load_state_dict(saved["optimizer"])
        if args.lr is not None:
            for group in optimizer.param_groups:
                if group.get("name", "recognizer") == "recognizer":
                    group["lr"] = args.lr
        if learned and getattr(args, "threshold_lr", None) is not None:
            optimizer.param_groups[1]["lr"] = args.threshold_lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=selection_mode, factor=0.5,
        patience=getattr(args, "lr_patience", 3), min_lr=1e-6)
    if restoring_optimizer and comparable and "scheduler" in saved:
        scheduler.load_state_dict(saved["scheduler"])
    if "rng" in saved:
        restore_rng(saved["rng"], device)
    start_epoch = int(saved.get("epoch", 0))
    initial_best = float("inf") if selection_mode == "min" else -1.0
    fallback_best = saved.get("validation", {}).get(metric, initial_best) if selection_mode == "min" else saved.get("best_f1", initial_best)
    best_score = float(saved.get("best_score", fallback_best)) if comparable else initial_best
    best_f1 = float(saved.get("best_f1", -1)) if comparable else -1.0
    stale = int(saved.get("stale_epochs", 0)) if comparable else 0
    if getattr(args, "reset_early_stopping", False):
        stale = 0
        scheduler.num_bad_epochs = 0
        print("Reset non-improvement and scheduler counts; keeping the previous best-score target", flush=True)
    if getattr(args, "patience", 0):
        print("Ignoring legacy --patience; early stopping has been removed", flush=True)

    def checkpoint_payload(epoch, metrics, training_metrics=None):
        chosen = saved_thresholds(metrics)
        payload = {"format_version": checkpoint_format_version(architecture), "model": model.state_dict(), "model_config": model_config(model),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "validation": metrics, "best_f1": best_f1, "best_score": best_score,
                "selection_metric": metric, "selection_mode": selection_mode, "stale_epochs": stale, "threshold": metrics["threshold"],
                "thresholds": chosen.to_dict(), "min_note_seconds": minimum, "validation_signature": signature,
                "sample_rate": SAMPLE_RATE, "hop_length": HOP_LENGTH, "rng": rng_state(),
                "training_config": {"augment": augment, "positive_weight": positive_weight,
                    "event_weight": event_weight, "threshold_grid": [] if learned else [values.to_dict() for values in grid],
                    "thresholds_only": thresholds_only, "freeze_thresholds": freeze_thresholds, **threshold_options,
                    "threshold_objective": "pitch_f0123" if fourier else "pitch_f03" if balanced else "f1",
                    "seconds": args.seconds, "windows_per_file": args.windows_per_file, "seed": seed,
                    "early_stopping": False, "patience": 0, "threshold_calibration": calibration_mode,
                    "calibration_values": calibration_values, "calibration_every": calibration_every,
                    "pitch_emphasis_version": PITCH_EMPHASIS_VERSION, **training_options}}
        if training_metrics is not None:
            payload["training_metrics"] = training_metrics
        if fourier:
            payload["model_version"] = FOURIER_MODEL_VERSION
            payload["training_config"]["model_version"] = FOURIER_MODEL_VERSION
        return payload

    def validation_score(epoch=None):
        started = time.perf_counter()
        candidates = grid
        search = "grid" if grid is not None else "parameters"
        if validation_calibration:
            current = model.learned_thresholds()
            full_search = (epoch is None or epoch == 1 or epoch % calibration_every == 0
                           or epoch == start_epoch + args.epochs)
            search = "full" if full_search else "current"
            candidates = [current]
            if full_search:
                candidates = list(dict.fromkeys([current, *threshold_grid(
                    current, calibration_values, calibration_values, calibration_values)]))
        metrics = score(model, val_loader, device, selection_metric=metric,
                        min_note_seconds=minimum, threshold_configs=candidates,
                        **({"avgf_temperature": threshold_options["threshold_temperature"],
                            "avgf_regularization": threshold_options["threshold_regularization"]} if fourier else {}))
        if validation_calibration and saved_thresholds(metrics) != current:
            model.threshold_module.initialize(saved_thresholds(metrics))
        metrics.update(calibration_search=search,
                       calibration_candidates=len(candidates) if candidates is not None else 1,
                       validation_seconds=time.perf_counter() - started)
        return metrics

    if resume:
        print(f"Restored {architecture} model at epoch {start_epoch}; optimizer={'restored' if restoring_optimizer else 'fresh'}", flush=True)
    elif not init_from:
        print(f"Fresh random initialization; seed={seed}; starting at epoch 1", flush=True)
    version_label = f"V{FOURIER_MODEL_VERSION} " if fourier else ""
    print(f"Training {version_label}{architecture} ({sum(p.numel() for p in model.parameters()):,} parameters) on "
          f"{len(train_set.records)} recordings; validating on {len(val_set.records)}; device={device}; "
          f"selection={metric}; epochs={args.epochs}; early_stopping=disabled; augmentation={augment}", flush=True)
    print((f"Calibrating three global threshold parameters on decoded validation {metric} every {calibration_every} epochs; validating current thresholds every epoch" if validation_calibration else
           "Keeping learned threshold parameters fixed for this run" if freeze_thresholds else
           f"Learning {model.threshold_module.raw.numel()} {'global' if global_thresholds else 'register'} thresholds on training targets")
          if learned else f"Calibrating {len(grid)} threshold combinations on validation windows", flush=True)
    if training_options["bass_sampling"]:
        print(f"Bass-focused window fraction={training_options['bass_sampling']:.2f}; "
              f"{len(train_set.bass_anchors)} training bass strikes indexed; validation sampling unchanged", flush=True)
    if training_options["treble_sampling"]:
        print(f"Treble-focused window fraction={training_options['treble_sampling']:.2f}; "
              f"{len(train_set.treble_anchors)} training treble strikes indexed; validation sampling unchanged", flush=True)
    if training_options["middle_sampling"]:
        print(f"Middle-focused window fraction={training_options['middle_sampling']:.2f}; "
              f"MIDI {train_set.middle_min_note}-{train_set.middle_max_note}; "
              f"{len(train_set.middle_anchors)} training middle strikes indexed; validation sampling unchanged", flush=True)
    if not any(training_options[name] for name in ("bass_sampling", "treble_sampling", "middle_sampling")):
        print("Training windows use ordinary random sampling across recordings", flush=True)
    if resume and reset_pitch_emphasis and any(prior.get(name, default) != default
                                               for name, default in pitch_emphasis_defaults.items()):
        print("Cleared saved pitch emphasis; explicit CLI pitch controls override the equal-note defaults", flush=True)
    if balanced:
        threshold_scores = "F0,F1,F2,F3" if fourier else "F0,F3"
        weight_summary = ("Equal error weights for all 88 piano keys; " if pitch_weights is None else
                          f"Bass/treble error weight={training_options['edge_loss_weight']:g}; "
                          f"middle error weight={training_options['middle_loss_weight']:g}; ")
        print(weight_summary +
              f"threshold objective=smooth per-pitch mean({threshold_scores}); validation uses decoded notes", flush=True)
    if not comparable and (resume or init_from and complete_transfer):
        initial_metrics = validation_score()
        best_score, best_f1 = initial_metrics[metric], initial_metrics["f1"]
        save_checkpoint(checkpoint_payload(start_epoch, initial_metrics), output)
        print(f"Saved starting model for comparison: {metric}={best_score:.4f}; "
              "best checkpoint will change only on validation improvement", flush=True)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        if thresholds_only:
            model.eval()  # Freeze dropout and batch-normalization statistics as well as weights.
        # Accumulate detached scalars on the device; transfer only once per epoch.
        running = torch.zeros(2 + len(LOSS_COMPONENTS), dtype=torch.float64, device=device)
        frame_loss = _FrameBCEAccumulator()
        for waves, targets in train_loader:
            waves = waves.to(device)
            targets = {key: value.to(device) for key, value in targets.items() if isinstance(value, torch.Tensor)}
            optimizer.zero_grad(set_to_none=True)
            predictions = model(waves)
            components = training_loss_components(predictions, targets, positive_weight, event_weight,
                                                 training_options["offset_loss_weight"], training_options["release_loss_weight"], pitch_weights)
            loss = sum(components.values())
            threshold_loss = loss.new_zeros(())
            components["threshold"] = loss.new_zeros(())
            if learned and not freeze_thresholds and not validation_calibration:
                threshold_loss = model.threshold_module.loss(predictions, targets,
                    threshold_options["threshold_temperature"], threshold_options["threshold_regularization"],
                    **({"pitch_weights": pitch_weights} if balanced else {}))
                components["threshold"] = threshold_options["threshold_loss_weight"] * threshold_loss
                loss = loss + components["threshold"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += torch.stack([loss, threshold_loss, *components.values()]).detach().to(torch.float64)
            frame_loss.add(predictions["frame"] if isinstance(predictions, dict) else predictions,
                           targets["frame"])
        averages = (running / len(train_loader)).cpu().tolist()
        training_metrics = {"total_loss": averages[0], "frame_bce": frame_loss.mean(),
                            "loss_components": dict(zip(LOSS_COMPONENTS, averages[2:]))}
        train_seconds = time.perf_counter() - epoch_started
        metrics = validation_score(epoch)
        chosen = saved_thresholds(metrics)
        value = metrics[metric]
        improved = value < best_score - 1e-6 if selection_mode == "min" else value > best_score + 1e-6
        stale = 0 if improved else stale + 1
        if improved:
            best_score = value
        best_f1 = max(best_f1, metrics["f1"])
        scheduler.step(value)
        log = {"epoch": epoch, "train_loss": training_metrics["total_loss"],
               "train_frame_bce": training_metrics["frame_bce"],
               "train_loss_components": training_metrics["loss_components"], "validation": metrics,
               "lr": optimizer.param_groups[0]["lr"], "selection_metric": metric, "selection_mode": selection_mode, "seed": seed,
               "best_score": best_score, "stale_epochs": stale, "early_stopping": False, "patience": 0,
               "train_seconds": train_seconds, "validation_seconds": metrics["validation_seconds"],
               "epoch_seconds": time.perf_counter() - epoch_started,
               "calibration_search": metrics["calibration_search"],
               "calibration_candidates": metrics["calibration_candidates"]}
        if fourier:
            log["model_version"] = FOURIER_MODEL_VERSION
        if learned:
            log.update(threshold_loss=averages[1],
                       threshold_lr=optimizer.param_groups[1]["lr"])
        summary = chosen.summary()
        pitch_summary = (f"note_macro_f1={metrics['note_macro_f1']:.4f} "
                         f"covered_pitches={metrics['note_macro_pitch_count']}/{N_NOTES} ") if "note_macro_f1" in metrics else ""
        if balanced and "note_macro_f03" in metrics:
            pitch_summary += (f"note_macro_f0={metrics['note_macro_f0']:.4f} "
                              f"note_macro_f3={metrics['note_macro_f3']:.4f} "
                              f"note_macro_f03={metrics['note_macro_f03']:.4f} ")
        if fourier and "note_macro_f0123" in metrics:
            pitch_summary += (f"note_macro_f2={metrics['note_macro_f2']:.4f} "
                              f"note_macro_f0123={metrics['note_macro_f0123']:.4f} ")
        threshold_digits = 6 if balanced else 2
        component_summary = " ".join(f"{name}={value:.4f}" for name, value in log["train_loss_components"].items())
        avgf_summary = f"val_avgf_loss={metrics['avgf_loss']:.6f} " if "avgf_loss" in metrics else ""
        print(f"epoch={epoch} train_total_loss={log['train_loss']:.4f} "
              f"train_frame_bce={log['train_frame_bce']:.4f} val_frame_bce={metrics['frame_bce']:.4f} "
              f"{avgf_summary}val_f1={metrics['f1']:.4f} onset_f1={metrics['onset_f1']:.4f} note_f1={metrics['note_f1']:.4f} "
              f"note_f_avg={metrics['note_f_avg']:.4f} stale_epochs={stale} "
              f"{pitch_summary}"
              f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
              f"frame={summary['frame']:.{threshold_digits}f} onset={summary['onset']:.{threshold_digits}f} offset={summary['offset']:.{threshold_digits}f} lr={log['lr']:.6g} "
              f"calibration={log['calibration_search']} threshold_sets={log['calibration_candidates']} "
              f"train_s={train_seconds:.1f} val_s={log['validation_seconds']:.1f}", flush=True)
        print(f"  train_loss_components: {component_summary}", flush=True)
        payload = checkpoint_payload(epoch, metrics, training_metrics)
        save_checkpoint(payload, latest)
        if improved:
            save_checkpoint(payload, output)
            print(f"Saved best model: {output}", flush=True)
        with output.with_suffix(".history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log) + "\n")
    print(f"Latest training state: {latest}", flush=True)
