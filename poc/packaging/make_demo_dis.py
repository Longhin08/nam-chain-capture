"""Prepara os DIs de exemplo (CC0, Freesound) em app/assets/di/.

    python packaging/make_demo_dis.py <pasta com os MP3 HQ do Freesound>

Fontes (todas "Creative Commons 0"; baixadas das prévias HQ públicas):
- 613906 "clean guitar loop.wav" — mil_dew — Strat em DI, 71 BPM
- 453384 "Clean Strumming & Arpeggio" — loudernoises — Focusrite, sem processamento
- 182803 "Guitar Jamming" — unfa — Explorer drop-D, sem processamento (trecho de 30 s).
  O autor pede que seus sons não sejam usados para treinar modelos generativos:
  no app este arquivo é só para ouvir o preview, nunca entra em treino.

Processamento: mono, 48 kHz, corte de silêncio nas pontas, pico em -6 dBFS
(nível típico de DI), fades de 10 ms. Nada de EQ/compressão.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48_000
OUT = Path(__file__).resolve().parents[1] / "app" / "assets" / "di"

SOURCES = [
    # (arquivo, nome no app, trecho (início, fim) em s ou None)
    ("613906_13499242-hq.mp3", "Strat limpa - loop (mil_dew)", None),
    ("453384_5859881-hq.mp3", "Dedilhado e acordes (loudernoises)", None),
    ("182803_1038806-hq.mp3", "Jam com licks e acordes (unfa)", "auto30"),
]


def to_mono_48k(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype(np.float32)


def trim_silence(x: np.ndarray, thresh_db: float = -50) -> np.ndarray:
    env = np.sqrt(np.convolve(x**2, np.ones(480) / 480, "same"))
    idx = np.where(env > 10 ** (thresh_db / 20) * np.max(np.abs(x)))[0]
    a, b = max(idx[0] - 480, 0), min(idx[-1] + 4800, len(x))
    return x[a:b]


def best_window(x: np.ndarray, seconds: float = 30) -> np.ndarray:
    """Janela com mais ataques (notas sendo tocadas) e sem trechos de silêncio."""
    hop = SR // 100
    env = np.sqrt(np.convolve(x**2, np.ones(hop) / hop, "same"))[::hop]
    db = 20 * np.log10(env + 1e-9)
    rise = np.zeros_like(db)
    rise[3:] = db[3:] - db[:-3]
    onsets = (rise > 6) & (db > db.max() - 40)
    w = int(seconds * 100)
    best, best_s = -1, 0
    for s in range(0, len(db) - w, 100):
        seg = db[s : s + w]
        if np.mean(seg < db.max() - 45) > 0.05:  # evita pausas longas
            continue
        score = onsets[s : s + w].sum()
        if score > best:
            best, best_s = score, s
    a = best_s * hop
    return x[a : a + int(seconds * SR)]


def finish(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    x = x * (10 ** (-6 / 20) / np.max(np.abs(x)))
    f = int(0.01 * SR)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    return x.astype(np.float32)


def main(src_dir: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, name, span in SOURCES:
        x, sr = sf.read(str(Path(src_dir) / fname), dtype="float32")
        x = to_mono_48k(x, sr)
        x = best_window(trim_silence(x)) if span == "auto30" else trim_silence(x)
        x = finish(x)
        dest = OUT / f"{name}.wav"
        sf.write(str(dest), x, SR, subtype="PCM_16")
        print(f"{dest.name}: {len(x) / SR:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1])
