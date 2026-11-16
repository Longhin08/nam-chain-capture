"""IR de caixa genérica (4x12 fechada), gerado a partir de uma curva de resposta.

Capturas "só amp" soam ásperas sem uma caixa depois. Para o app ter um IR que
pode ser distribuído livremente, geramos um a partir de uma curva típica de
alto-falante de guitarra (não é medição de nenhum produto):

- corte de graves ~75 Hz com ressonância da caixa ~110 Hz
- leve vale nos médios-graves (~400 Hz)
- pico de presença ~2,5 kHz e vale ~4,2 kHz
- queda forte acima de ~5 kHz (o que "tira o zumbido" do amp)

O IR é de fase mínima (causal, sem pré-eco), 20 ms, 48 kHz.
"""

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np

SAMPLE_RATE = 48_000
LENGTH = 960  # 20 ms: dentro do limite em que o modelo final ainda captura a cauda

# (frequência Hz, ganho dB) — interpolado em escala log.
CURVE_4X12: Sequence[Tuple[float, float]] = (
    (20, -36), (40, -20), (60, -8), (75, -2), (110, 3.5), (160, 1.5),
    (250, 0), (400, -3), (700, -1), (1200, 0), (1800, 2.5), (2500, 5),
    (3300, 2), (4200, -6), (5000, -9), (6000, -20), (8000, -34),
    (12000, -48), (24000, -60),
)


def _min_phase_from_magnitude(mag: np.ndarray) -> np.ndarray:
    """Espectro de fase mínima (método do cepstro real) a partir de |H| em rfft."""
    n = 2 * (len(mag) - 1)
    log_mag = np.log(np.maximum(mag, 1e-9))
    cep = np.fft.irfft(log_mag, n)
    fold = np.zeros(n)
    fold[0] = cep[0]
    fold[1 : n // 2] = 2 * cep[1 : n // 2]
    fold[n // 2] = cep[n // 2]
    return np.exp(np.fft.rfft(fold, n))


def make_cab_ir(
    curve: Sequence[Tuple[float, float]] = CURVE_4X12,
    length: int = LENGTH,
    sr: int = SAMPLE_RATE,
    nfft: int = 16384,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    f_pts = np.array([c[0] for c in curve], dtype=float)
    g_pts = np.array([c[1] for c in curve], dtype=float)
    g = np.interp(np.log10(np.maximum(freqs, f_pts[0])), np.log10(f_pts), g_pts)
    h = np.fft.irfft(_min_phase_from_magnitude(10 ** (g / 20)), nfft)[:length]
    # Janela de saída suave nos últimos 25% para não "cortar" a cauda.
    fade = int(length * 0.25)
    h[-fade:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))
    h = h / np.sqrt(np.sum(h**2))  # energia unitária (como o load_ir normaliza)
    return h.astype(np.float32)


def write_cab_ir(path: Union[str, Path]) -> Path:
    from scipy.io import wavfile

    path = Path(path)
    h = make_cab_ir()
    wavfile.write(str(path), SAMPLE_RATE, (h / np.max(np.abs(h)) * 0.9).astype(np.float32))
    return path
