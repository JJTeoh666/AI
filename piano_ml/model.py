"""PyTorch framewise 88-note classifier."""

import torch
import sys
from contextlib import nullcontext
from torch import nn
import torchaudio
from .learned_thresholds import (GlobalThresholds, PitchBalancedThresholds,
                                 PitchFourScoreThresholds, RegisterThresholds)

SAMPLE_RATE = 16000
HOP_LENGTH = 320
LOW_NOTE = 21
HIGH_NOTE = 108
N_NOTES = HIGH_NOTE - LOW_NOTE + 1
V7_ARCHITECTURE = "onsets-multires-balanced"
FOURIER_ARCHITECTURE = "onsets-fourier-recurrent"
FOURIER_MODEL_VERSION = "8.1"
DEFAULT_ARCHITECTURE = FOURIER_ARCHITECTURE
BALANCED_ARCHITECTURES = (V7_ARCHITECTURE, FOURIER_ARCHITECTURE)
RECURRENT_CONV_ARCHITECTURES = ("onsets-recurrent", "onsets-recurrent-global",
                               "onsets-multires-global", V7_ARCHITECTURE)
MULTIRES_ARCHITECTURES = ("onsets-multires-global", *BALANCED_ARCHITECTURES)


def default_selection_metric(architecture, model_version=None):
    if architecture == FOURIER_ARCHITECTURE:
        return "note_macro_f0123" if model_version == "8" else "avgf_loss"
    if architecture == V7_ARCHITECTURE:
        return "note_macro_f03"
    return "f1" if architecture == "frame" else "note_f_avg"


def checkpoint_format_version(architecture):
    return {"frame": 4, "onsets": 4, "onsets-calibrated": 5, "onsets-recurrent": 5,
            "onsets-recurrent-global": 6, "onsets-multires-global": 7,
            V7_ARCHITECTURE: 8, FOURIER_ARCHITECTURE: 9}[architecture]


class PianoNet(nn.Module):
    architecture = "frame"

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=2048, win_length=2048,
            hop_length=HOP_LENGTH, n_mels=128, f_min=27.5, f_max=8000.0,
            power=2.0, center=True,
        )
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        # Preserve absolute frequency position: averaging it away makes pitch
        # classification nearly impossible for a convolutional model.
        self.head = nn.Conv1d(64 * 16, N_NOTES, 1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # Input [batch, samples]; output [batch, frames, 88] logits.
        mel = torch.log1p(self.mel(wave).clamp_min(0))
        x = self.features(mel.unsqueeze(1)).flatten(1, 2)
        return self.head(x).transpose(1, 2)


class PortableGRU(nn.GRU):
    """Allow native CUDA recurrence without loading cuDNN RNN descriptors."""

    def __init__(self, *args, use_cudnn: bool = True, **kwargs):
        self.use_cudnn = use_cudnn
        super().__init__(*args, **kwargs)

    def flatten_parameters(self):
        if self.use_cudnn:
            super().flatten_parameters()

    def forward(self, sequence, hx=None):
        context = torch.backends.cudnn.flags(enabled=False) if sequence.is_cuda and not self.use_cudnn else nullcontext()
        with context:
            return super().forward(sequence, hx)


class OnsetsPianoNet(PianoNet):
    """Offline CNN + bidirectional GRU with onset-conditioned note frames."""

    architecture = "onsets"

    def __init__(self, hidden_size: int = 128, gru_layers: int = 2,
                 dropout: float = 0.2, rnn_backend: str = "auto") -> None:
        if hidden_size < 1 or gru_layers < 1 or rnn_backend not in ("auto", "native", "cudnn"):
            raise ValueError("GRU size/layer count must be positive and backend auto, native or cudnn.")
        super().__init__()
        del self.head
        self.config = {"architecture": self.architecture, "hidden_size": hidden_size,
                       "gru_layers": gru_layers, "dropout": dropout, "rnn_backend": rnn_backend}
        self.projection = nn.Sequential(nn.Linear(64 * 16, 256), nn.LayerNorm(256),
                                        nn.ReLU(), nn.Dropout(dropout))
        # The installed Windows nightly completes cuDNN GRU computation but
        # exits abnormally on shutdown. Native recurrence passes that check.
        windows_nightly = sys.platform == "win32" and ".dev" in torch.__version__
        use_cudnn = rnn_backend == "cudnn" or (rnn_backend == "auto" and not windows_nightly)
        self.temporal = PortableGRU(256, hidden_size, num_layers=gru_layers, use_cudnn=use_cudnn,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if gru_layers > 1 else 0)
        width = hidden_size * 2
        self.onset_head = nn.Linear(width, N_NOTES)
        self.frame_head = nn.Linear(width + N_NOTES, N_NOTES)
        self.offset_head = nn.Linear(width, N_NOTES)
        self.velocity_head = nn.Linear(width, N_NOTES)
        self.pedal_head = nn.Linear(width, 1)

    def forward(self, wave: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence, _ = self.temporal(self.acoustic_features(wave))
        return self.predict_heads(sequence)

    def acoustic_features(self, wave):
        mel = torch.log1p(self.mel(wave).clamp_min(0))
        features = self.extract_features(mel.unsqueeze(1)).flatten(1, 2).transpose(1, 2)
        return self.projection(features)

    def predict_heads(self, sequence):
        onset = self.onset_head(sequence)
        frame = self.frame_head(torch.cat((sequence, onset.sigmoid()), dim=-1))
        return {"frame": frame, "onset": onset, "offset": self.offset_head(sequence),
                "velocity": self.velocity_head(sequence), "pedal": self.pedal_head(sequence)}

    def extract_features(self, mel):
        return self.features(mel)


def model_config(model: nn.Module) -> dict:
    return getattr(model, "config", {"architecture": "frame"})


class CalibratedPianoNet(OnsetsPianoNet):
    """Onset model expanded with learned frame/onset/offset register decisions."""

    architecture = "onsets-calibrated"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold_module = RegisterThresholds()

    def learned_thresholds(self):
        return self.threshold_module.export()


class RecurrentConvBlock(nn.Module):
    """Refine a frequency/time feature map with shared convolutional feedback."""

    def __init__(self, channels: int, steps: int = 3):
        super().__init__()
        if steps < 2:
            raise ValueError("Recurrent convolution requires at least two refinement steps.")
        self.steps = steps
        self.drive = nn.Conv2d(channels, channels, 3, padding=1)
        self.feedback = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        # An old checkpoint initially produces the same predictions. The scale
        # learns first; feedback weights receive gradients once it moves from zero.
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        drive = self.drive(features)
        hidden = torch.zeros_like(features)
        for _ in range(self.steps):
            hidden = torch.relu(self.norm(drive + self.feedback(hidden)))
        return features + self.scale.tanh() * hidden


class RecurrentPianoNet(CalibratedPianoNet):
    """Recurrent convolution at two CNN stages, followed by the sequence GRU."""

    architecture = "onsets-recurrent"

    def __init__(self, conv_steps: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.config["conv_steps"] = conv_steps
        self.recurrent_blocks = nn.ModuleDict({
            "stage2": RecurrentConvBlock(32, conv_steps),
            "stage4": RecurrentConvBlock(64, conv_steps),
        })

    def extract_features(self, mel):
        for index, layer in enumerate(self.features):
            mel = layer(mel)
            if index == 7:  # Second pool: [batch, 32 channels, 32 frequencies, time].
                mel = self.recurrent_blocks["stage2"](mel)
            elif index == 14:  # Final CNN activation: [batch, 64, 16, time].
                mel = self.recurrent_blocks["stage4"](mel)
        return mel


class GlobalRecurrentPianoNet(RecurrentPianoNet):
    """Version 5: recurrent CNN + GRU with three global learned thresholds."""

    architecture = "onsets-recurrent-global"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold_module = GlobalThresholds()
        # Fresh training activates both recurrence branches immediately.
        for block in self.recurrent_blocks.values():
            nn.init.uniform_(block.scale, -0.05, 0.05)


class MultiResolutionPianoNet(GlobalRecurrentPianoNet):
    """Version 6: preserve strike timing and add detailed low-frequency features."""

    architecture = "onsets-multires-global"

    def __init__(self, long_fft=8192, release_frames=3, **kwargs):
        if long_fft < 2048 or long_fft & (long_fft - 1):
            raise ValueError("Long FFT must be a power of two, at least 2048.")
        if not isinstance(release_frames, int) or release_frames < 2:
            raise ValueError("Release persistence must be at least two frames.")
        super().__init__(**kwargs)
        self.config.update(long_fft=long_fft, release_frames=release_frames)
        self.long_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=long_fft, win_length=long_fft,
            hop_length=HOP_LENGTH, n_mels=256, f_min=20.0, f_max=2000.0,
            power=2.0, center=True, pad_mode="constant")
        self.long_features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 1)))
        self.long_projection = nn.Sequential(nn.Linear(32 * 32, 256), nn.LayerNorm(256),
                                             nn.ReLU(), nn.Dropout(self.config["dropout"]))
        self.long_scale = nn.Parameter(torch.empty(()).uniform_(0.01, 0.05))
        # Releases can depend on note activity, strikes and pedal state.
        width = self.config["hidden_size"] * 2
        self.offset_refinement = nn.Sequential(nn.Linear(width + 2 * N_NOTES + 1, 128),
                                               nn.ReLU(), nn.Linear(128, N_NOTES))
        # A v5 transfer initially retains its exact recognizer predictions.
        nn.init.zeros_(self.offset_refinement[-1].weight)
        nn.init.zeros_(self.offset_refinement[-1].bias)

    def acoustic_features(self, wave):
        short = super().acoustic_features(wave)
        mel = torch.log1p(self.long_mel(wave).clamp_min(0))
        long = self.long_features(mel.unsqueeze(1)).flatten(1, 2).transpose(1, 2)
        return short + self.long_scale.tanh() * self.long_projection(long)

    def predict_heads(self, sequence):
        output = super().predict_heads(sequence)
        context = torch.cat((sequence, output["frame"].sigmoid(), output["onset"].sigmoid(),
                             output["pedal"].sigmoid()), dim=-1)
        output["offset"] = output["offset"] + self.offset_refinement(context)
        return output


class BalancedPianoNet(MultiResolutionPianoNet):
    """Version 7: multi-resolution recognizer with pitch-balanced gradient cutoffs."""

    architecture = V7_ARCHITECTURE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold_module = PitchBalancedThresholds()


class FourierMixing(nn.Module):
    """Learn channel mixing in temporal Fourier space, with a real DC matrix."""

    def __init__(self, width, modes=9, padding=8):
        super().__init__()
        if width < 1 or modes < 1 or padding < 0:
            raise ValueError("Fourier width/modes must be positive and padding nonnegative.")
        self.modes, self.padding = modes, padding
        self.dc_weight = nn.Parameter(torch.randn(width, width) / width ** 0.5)
        # Store real/imaginary parts as real parameters for portable AdamW state.
        # DC has no imaginary component, so every default parameter is usable.
        self.band_weight = nn.Parameter(torch.randn(modes - 1, width, width, 2)
                                        / (2 * width) ** 0.5)

    def forward(self, sequence):
        length = sequence.shape[1]
        if length < 1:
            raise ValueError("Fourier mixing needs at least one time frame.")
        original_dtype = sequence.dtype
        # FFT supports arbitrary clip lengths in float32 on CUDA.
        if sequence.dtype in (torch.float16, torch.bfloat16):
            sequence = sequence.float()
        if self.padding:
            sequence = nn.functional.pad(sequence.transpose(1, 2),
                                         (self.padding, self.padding)).transpose(1, 2)
        spectrum = torch.fft.rfft(sequence, dim=1, norm="ortho")
        modes = min(self.modes, spectrum.shape[1])
        mixed = torch.zeros_like(spectrum)
        mixed[:, 0] = (spectrum[:, 0].real @ self.dc_weight.to(sequence.dtype)).to(spectrum.dtype)
        if modes > 1:
            weights = torch.view_as_complex(self.band_weight[:modes - 1].to(sequence.dtype).contiguous())
            mixed[:, 1:modes] = torch.einsum("bmi,mio->bmo", spectrum[:, 1:modes], weights)
        result = torch.fft.irfft(mixed, n=sequence.shape[1], dim=1, norm="ortho")
        return result[:, self.padding:self.padding + length].to(original_dtype)


class FourierAnalysisLayer(nn.Module):
    """FAN projection: concatenate learned cosine, sine and aperiodic features."""

    def __init__(self, input_width, output_width):
        super().__init__()
        if input_width < 1 or output_width < 4:
            raise ValueError("FAN input width must be positive and output width at least four.")
        periodic = output_width // 4
        self.periodic = nn.Linear(input_width, periodic, bias=False)
        self.aperiodic = nn.Linear(input_width, output_width - 2 * periodic)

    def forward(self, features):
        phase = self.periodic(features)
        return torch.cat((phase.cos(), phase.sin(),
                          nn.functional.gelu(self.aperiodic(features))), dim=-1)


class FourierTemporalBlock(nn.Module):
    """Combine global Fourier mixing, local time convolution and residual FAN."""

    def __init__(self, width, modes=9, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.spectral = FourierMixing(width, modes)
        self.local = nn.Conv1d(width, width, 3, padding=1)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(FourierAnalysisLayer(width, width), nn.Dropout(dropout),
                                 nn.Linear(width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        normalized = self.norm(sequence)
        local = self.local(normalized.transpose(1, 2)).transpose(1, 2)
        sequence = sequence + self.dropout(nn.functional.gelu(self.spectral(normalized) + local))
        return sequence + self.dropout(self.mlp(self.mlp_norm(sequence)))


class FourierSpectralEncoder(nn.Module):
    """Keep absolute frequency positions while pooling only the frequency axis."""

    def __init__(self, bands, width, dropout=0.2):
        super().__init__()
        layers = []
        previous = 1
        for stage, channels in enumerate((16, 32, 64, 128)):
            layers.extend((nn.Conv2d(previous, channels, 3, padding=1),
                           nn.GroupNorm(8, channels), nn.GELU()))
            if stage < 3 or bands == 256:
                layers.append(nn.MaxPool2d((2, 1)))
            previous = channels
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Sequential(nn.Linear(128 * 16, width), nn.LayerNorm(width),
                                        nn.GELU(), nn.Dropout(dropout))

    def forward(self, mel):
        features = self.convolution(mel.unsqueeze(1)).flatten(1, 2).transpose(1, 2)
        return self.projection(features)


class FourierRecurrentPianoNet(nn.Module):
    """V8/V8.1: dual FFT/CNN encoders, Fourier Analysis Network layers and BiGRU."""

    architecture = FOURIER_ARCHITECTURE
    model_version = FOURIER_MODEL_VERSION

    def __init__(self, feature_width=384, fourier_modes=9, fourier_layers=2,
                 hidden_size=192, gru_layers=2, dropout=0.2, rnn_backend="auto",
                 long_fft=8192, release_frames=3):
        super().__init__()
        sizes = (feature_width, fourier_modes, fourier_layers, hidden_size, gru_layers)
        if any(not isinstance(value, int) or value < 1 for value in sizes):
            raise ValueError("Fourier/GRU dimensions and layer counts must be positive integers.")
        if feature_width < 4:
            raise ValueError("FAN feature width must be at least four.")
        if not isinstance(long_fft, int) or long_fft < 2048 or long_fft & (long_fft - 1):
            raise ValueError("Long FFT must be a power of two, at least 2048.")
        if not isinstance(release_frames, int) or release_frames < 2:
            raise ValueError("Release persistence must be at least two frames.")
        if not 0 <= dropout < 1 or rnn_backend not in ("auto", "native", "cudnn"):
            raise ValueError("Dropout must be inside [0,1); RNN backend must be auto, native or cudnn.")
        self.config = dict(architecture=self.architecture, feature_width=feature_width,
                           fourier_modes=fourier_modes, fourier_layers=fourier_layers,
                           hidden_size=hidden_size, gru_layers=gru_layers, dropout=dropout,
                           rnn_backend=rnn_backend, long_fft=long_fft, release_frames=release_frames)
        self.short_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=2048, win_length=2048, hop_length=HOP_LENGTH,
            n_mels=128, f_min=20.0, f_max=8000.0, center=True, pad_mode="constant")
        self.long_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=long_fft, win_length=long_fft, hop_length=HOP_LENGTH,
            n_mels=256, f_min=20.0, f_max=8000.0, center=True, pad_mode="constant")
        self.short_encoder = FourierSpectralEncoder(128, feature_width, dropout)
        self.long_encoder = FourierSpectralEncoder(256, feature_width, dropout)
        self.fusion = nn.Sequential(FourierAnalysisLayer(2 * feature_width, feature_width),
                                    nn.LayerNorm(feature_width), nn.Dropout(dropout))
        self.fourier = nn.Sequential(*[FourierTemporalBlock(feature_width, fourier_modes, dropout)
                                      for _ in range(fourier_layers)])
        windows_nightly = sys.platform == "win32" and ".dev" in torch.__version__
        use_cudnn = rnn_backend == "cudnn" or (rnn_backend == "auto" and not windows_nightly)
        self.temporal = PortableGRU(feature_width, hidden_size, num_layers=gru_layers,
                                   batch_first=True, bidirectional=True, use_cudnn=use_cudnn,
                                   dropout=dropout if gru_layers > 1 else 0)
        width = 2 * hidden_size
        self.sequence_norm = nn.LayerNorm(width)
        self.onset_head = nn.Linear(width, N_NOTES)
        self.frame_head = nn.Linear(width + N_NOTES, N_NOTES)
        self.offset_head = nn.Linear(width, N_NOTES)
        self.velocity_head = nn.Linear(width, N_NOTES)
        self.pedal_head = nn.Linear(width, 1)
        self.offset_refinement = nn.Sequential(nn.Linear(width + 2 * N_NOTES + 1, hidden_size),
                                               nn.GELU(), nn.Linear(hidden_size, N_NOTES))
        self.threshold_module = PitchFourScoreThresholds()

    def forward(self, wave):
        short = self.short_encoder(torch.log1p(self.short_mel(wave).clamp_min(0)))
        long = self.long_encoder(torch.log1p(self.long_mel(wave).clamp_min(0)))
        features = self.fourier(self.fusion(torch.cat((short, long), dim=-1)))
        sequence, _ = self.temporal(features)
        sequence = self.sequence_norm(sequence)
        onset = self.onset_head(sequence)
        frame = self.frame_head(torch.cat((sequence, onset.sigmoid()), dim=-1))
        pedal = self.pedal_head(sequence)
        context = torch.cat((sequence, frame.sigmoid(), onset.sigmoid(), pedal.sigmoid()), dim=-1)
        offset = self.offset_head(sequence) + self.offset_refinement(context)
        return {"frame": frame, "onset": onset, "offset": offset,
                "velocity": self.velocity_head(sequence), "pedal": pedal}

    def learned_thresholds(self):
        return self.threshold_module.export()


def build_model(config: dict | None = None) -> nn.Module:
    config = dict(config or {"architecture": "frame"})
    architecture = config.pop("architecture", "frame")
    if architecture == "frame":
        return PianoNet()
    if architecture == "onsets":
        return OnsetsPianoNet(**config)
    if architecture == "onsets-calibrated":
        return CalibratedPianoNet(**config)
    if architecture == "onsets-recurrent":
        return RecurrentPianoNet(**config)
    if architecture == "onsets-recurrent-global":
        return GlobalRecurrentPianoNet(**config)
    if architecture == "onsets-multires-global":
        return MultiResolutionPianoNet(**config)
    if architecture == V7_ARCHITECTURE:
        return BalancedPianoNet(**config)
    if architecture == FOURIER_ARCHITECTURE:
        return FourierRecurrentPianoNet(**config)
    raise ValueError(f"Unsupported model architecture: {architecture}")
