"""Torchlibrosa-compatible frozen frontend, restored entirely from the checkpoint.

The checkpoint contains both Hann-windowed DFT kernels and the Slaney mel bank.
Using those tensors directly preserves the original frontend without librosa,
torchlibrosa, or their historical dependency versions.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FrozenSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.conv_real = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop_length, bias=False)
        self.conv_imag = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop_length, bias=False)
        self.requires_grad_(False)

    def forward(self, wave):
        x = F.pad(wave[:, None], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        real, imag = self.conv_real(x), self.conv_imag(x)
        return real[:, None].transpose(2, 3), imag[:, None].transpose(2, 3)


class Spectrogram(nn.Module):
    def __init__(self, n_fft, hop_length, **kwargs):
        super().__init__()
        self.stft = FrozenSTFT(n_fft, hop_length)

    def forward(self, wave):
        real, imag = self.stft(wave)
        return real ** 2 + imag ** 2


class LogmelFilterBank(nn.Module):
    def __init__(self, n_fft, n_mels, amin=1e-10, **kwargs):
        super().__init__()
        self.melW = nn.Parameter(torch.empty(n_fft // 2 + 1, n_mels), requires_grad=False)
        self.amin = amin

    def forward(self, spectrogram):
        return 10.0 * torch.log10(torch.clamp(torch.matmul(spectrogram, self.melW), min=self.amin, max=float("inf")))
