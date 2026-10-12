"""Fixed, centered constant-Q filters and piano harmonic stacking in PyTorch.

Octave groups share an FFT, then project its coefficients onto variable-length
Hann filters. Negligible Fourier coefficients are omitted to avoid dense,
long time-domain convolutions. Every group uses the same 320-sample hop.
"""

import math

import torch
from torch import nn


SAMPLE_RATE = 16000
HOP_LENGTH = 320
BINS_PER_OCTAVE = 36
F_MIN = 27.5
PIANO_BINS = 88 * 3
HARMONICS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def cqt_basis(frequencies, n_fft):
    """L1-normalized Hann wavelets, centered at the FFT's middle sample."""
    quality = 1 / (2 ** (1 / BINS_PER_OCTAVE) - 1)
    basis = torch.zeros(len(frequencies), 2, n_fft, dtype=torch.float32)
    for index, frequency in enumerate(frequencies):
        length = math.ceil(quality * SAMPLE_RATE / frequency)
        window = torch.hann_window(length, periodic=True, dtype=torch.float64)
        window /= window.sum()
        phase = (torch.arange(length, dtype=torch.float64) - length // 2) * (2 * math.pi * frequency / SAMPLE_RATE)
        start = n_fft // 2 - length // 2
        basis[index, 0, start:start + length] = window * phase.cos()
        basis[index, 1, start:start + length] = -window * phase.sin()
    return basis


class CQTProjection(nn.Module):
    def __init__(self, frequencies):
        super().__init__()
        quality = 1 / (2 ** (1 / BINS_PER_OCTAVE) - 1)
        length = math.ceil(quality * SAMPLE_RATE / frequencies[0])
        self.n_fft = 1 << (length - 1).bit_length()
        self.n_bins = len(frequencies)
        spectrum = torch.fft.rfft(cqt_basis(frequencies, self.n_fft))
        # Bound discarded coefficients relative to each wavelet's peak, rather
        # than dropping weak bins merely because a different bin is stronger.
        peak = spectrum.abs().amax(dim=-1, keepdim=True)
        indices = torch.where((spectrum.abs() >= peak * 1e-5).any(dim=(0, 1)))[0]
        factor = torch.full((indices.numel(),), 2 / self.n_fft)
        factor[(indices == 0) | (indices == self.n_fft // 2)] = 1 / self.n_fft
        coefficients = torch.view_as_real(spectrum[..., indices]) * factor[None, None, :, None]
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("coefficients", coefficients.reshape(2 * self.n_bins, -1), persistent=False)
        self.register_buffer("window", torch.ones(self.n_fft), persistent=False)

    def forward(self, wave):
        spectrum = torch.stft(wave, n_fft=self.n_fft, hop_length=HOP_LENGTH,
                              window=self.window, center=True, pad_mode="constant",
                              return_complex=True)
        selected = torch.view_as_real(spectrum[:, self.indices].transpose(1, 2)).flatten(-2)
        projected = nn.functional.linear(selected, self.coefficients)
        magnitude = torch.linalg.vector_norm(projected.reshape(*projected.shape[:2], self.n_bins, 2), dim=-1)
        return magnitude.transpose(1, 2)


class ConstantQ(nn.Module):
    """A0 through the highest bin below Nyquist, with 3 bins per semitone."""

    def __init__(self):
        super().__init__()
        count = math.ceil(BINS_PER_OCTAVE * math.log2((SAMPLE_RATE / 2) / F_MIN))
        frequencies = [F_MIN * 2 ** (index / BINS_PER_OCTAVE) for index in range(count)]
        self.register_buffer("frequencies", torch.tensor(frequencies), persistent=False)
        self.groups = nn.ModuleList([CQTProjection(frequencies[first:first + BINS_PER_OCTAVE])
                                     for first in range(0, count, BINS_PER_OCTAVE)])

    def forward(self, wave):
        # FFT kernels remain float32 under mixed-precision model training.
        with torch.autocast(device_type=wave.device.type, enabled=False):
            return torch.cat([group(wave.float()) for group in self.groups], dim=1)


class HarmonicStack(nn.Module):
    """Align CQT energy at h*f with the candidate's fundamental frequency f."""

    def __init__(self, input_bins):
        super().__init__()
        shifts = [round(BINS_PER_OCTAVE * math.log2(harmonic)) for harmonic in HARMONICS]
        indices = torch.arange(PIANO_BINS)[None, :] + torch.tensor(shifts)[:, None]
        self.register_buffer("valid", (indices >= 0) & (indices < input_bins), persistent=False)
        self.register_buffer("indices", indices.clamp(0, input_bins - 1), persistent=False)

    def forward(self, cqt):
        return cqt[:, self.indices, :] * self.valid[None, :, :, None]
