"""Validated decoding thresholds and independent calibration grids."""

from dataclasses import asdict, dataclass
from itertools import product
import numpy as np

REGISTERS = (("bass", 21, 47), ("middle", 48, 83), ("treble", 84, 108))
HEADS = ("frame", "onset", "offset")


@dataclass(frozen=True)
class Thresholds:
    frame: float | tuple = 0.5
    onset: float | tuple = 0.5
    offset: float | tuple = 0.5

    def __post_init__(self):
        for name, value in asdict(self).items():
            array = np.asarray(value, dtype=float)
            if array.shape not in ((), (88,)) or not np.all((array > 0) & (array < 1)):
                raise ValueError(f"{name.capitalize()} threshold must be between 0 and 1.")
            object.__setattr__(self, name, float(array) if array.ndim == 0 else tuple(array.tolist()))

    def to_dict(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    def summary(self) -> dict:
        """Scalar means for legacy aliases and global manual controls."""
        return {key: float(np.mean(value)) for key, value in asdict(self).items()}

    def array(self, head: str) -> np.ndarray:
        return np.broadcast_to(np.asarray(getattr(self, head)), (88,))

    @property
    def pitch_dependent(self) -> bool:
        return any(isinstance(getattr(self, head), tuple) for head in HEADS)

    def registers(self) -> list[dict]:
        return [{"name": name, "low_note": low, "high_note": high,
                 **{head: float(self.array(head)[low - 21:high - 20].mean()) for head in HEADS}}
                for name, low, high in REGISTERS]

    def override(self, shared=None, frame=None, onset=None, offset=None):
        return Thresholds(frame=self.frame if shared is None and frame is None else (frame if frame is not None else shared),
                          onset=self.onset if shared is None and onset is None else (onset if onset is not None else shared),
                          offset=self.offset if offset is None else offset)


def saved_thresholds(saved: dict) -> Thresholds:
    """Old checkpoints shared frame/onset and always used offset 0.5."""
    if isinstance(saved.get("thresholds"), dict):
        return Thresholds(**saved["thresholds"])
    value = float(saved.get("threshold", 0.5))
    return Thresholds(frame=value, onset=value, offset=0.5)


def threshold_grid(base: Thresholds, frame_values=None, onset_values=None,
                   offset_values=None, shared_values=None) -> list[Thresholds]:
    if shared_values is not None:
        if any(values is not None for values in (frame_values, onset_values, offset_values)):
            raise ValueError("Use --thresholds for a shared sweep, or separate --frame/onset/offset-thresholds.")
        grid = [base.override(shared=value) for value in shared_values]
    else:
        axes = (frame_values if frame_values is not None else [base.frame],
                onset_values if onset_values is not None else [base.onset],
                offset_values if offset_values is not None else [base.offset])
        grid = [Thresholds(*values) for values in product(*axes)]
    if not grid:
        raise ValueError("Threshold grids cannot be empty.")
    return sorted(set(grid), key=lambda value: tuple(number for head in HEADS for number in value.array(head)))


def requested_grid(args, base: Thresholds, architecture: str,
                   training: bool = False, prior: dict | None = None) -> list[Thresholds]:
    """Resolve CLI grids, fixed overrides, fresh defaults, and resumed grids."""
    frame = getattr(args, "frame_thresholds", None)
    onset = getattr(args, "onset_thresholds", None)
    offset = getattr(args, "offset_thresholds", None)
    shared = getattr(args, "thresholds", None)
    if architecture == "frame" and (onset is not None or offset is not None
            or getattr(args, "onset_threshold", None) is not None
            or getattr(args, "offset_threshold", None) is not None):
        raise ValueError("Onset/offset thresholds require an onsets model; this checkpoint has only a frame head.")
    base = base.override(shared=getattr(args, "threshold", None),
                         frame=getattr(args, "frame_threshold", None),
                         onset=getattr(args, "onset_threshold", None),
                         offset=getattr(args, "offset_threshold", None))
    if training and all(values is None for values in (frame, onset, offset, shared)):
        if prior and "threshold_grid" in prior:
            return [Thresholds(**values) for values in prior["threshold_grid"]]
        frame = [0.35, 0.5, 0.65] if architecture != "frame" else [0.35, 0.45, 0.5, 0.55, 0.65, 0.75]
        if architecture != "frame":
            onset, offset = [0.3, 0.5, 0.7], [0.3, 0.5, 0.7]
    return threshold_grid(base, frame, onset, offset, shared)
