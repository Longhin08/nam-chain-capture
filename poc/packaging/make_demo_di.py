"""Gera app/assets/demo_di.wav: um riff original de guitarra DI (sem amp).

Usa as notas gravadas do banco "Electric Guitar FSBS (direct)" do FreePats,
CC0 (domínio público): guitarra Fender, captador da ponte, direto na interface.
https://freepats.zenvoid.org/ElectricGuitar/clean-electric-guitar.html

    python packaging/make_demo_di.py "<pasta extraída do EGuitarFSBS-direct SFZ+FLAC>"

O riff é composição própria deste projeto (Lá menor, ~96 BPM): acorde, frases
na pentatônica, acordes abertos e um trecho nas cordas graves, para o preview
mostrar ataque, sustentação, acordes e graves passando pela cadeia.
"""

import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48_000
BPM = 96
BEAT = 60 / BPM
OUT = Path(__file__).resolve().parents[1] / "app" / "assets" / "demo_di.wav"

NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def midi(name: str) -> int:
    m = re.fullmatch(r"([A-G]#?)(\d)", name)
    return 12 * (int(m.group(2)) + 1) + NOTE[m.group(1)]


# ------------------------------------------------------------- banco SFZ


def parse_sfz(path: Path):
    """Lista de grupos: {lokey, hikey, keycenter, soft(bool), samples[]}."""
    groups, cur = [], None
    for raw in path.read_text().splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        if line.startswith("<group>"):
            cur = {"samples": [], "soft": False}
            groups.append(cur)
            continue
        if cur is None or line.startswith("<"):
            continue
        for k, v in re.findall(r"(\w+)=(\S+)", line):
            if k == "lokey":
                cur["lokey"] = int(v)
            elif k == "hikey":
                cur["hikey"] = int(v)
            elif k == "key":
                cur["lokey"] = cur["hikey"] = cur["keycenter"] = int(v)
            elif k == "pitch_keycenter":
                cur["keycenter"] = int(v)
            elif k == "hivel":
                cur["soft"] = True
            elif k == "sample":
                cur["samples"].append(path.parent / v)
    return [g for g in groups if g["samples"] and "lokey" in g]


class Bank:
    def __init__(self, root: Path):
        sfz = next(root.glob("*.sfz"))
        self.groups = parse_sfz(sfz)
        self._cache = {}
        self._rr = {}

    def note(self, key: int, soft: bool) -> np.ndarray:
        cands = [g for g in self.groups if g["lokey"] <= key <= g["hikey"] and g["soft"] == soft]
        if not cands:
            raise ValueError(f"Sem amostra para a nota {key}")
        g = cands[0]
        i = self._rr.get(id(g), 0)  # alterna as gravações (round robin)
        self._rr[id(g)] = i + 1
        f = g["samples"][i % len(g["samples"])]
        if f not in self._cache:
            x, sr = sf.read(str(f), dtype="float32")
            assert sr == SR and x.ndim == 1, f
            self._cache[f] = x
        x = self._cache[f]
        semis = key - g["keycenter"]
        if semis:
            r = Fraction(2 ** (-semis / 12)).limit_denominator(400)  # mais agudo = mais curto
            x = resample_poly(x, r.numerator, r.denominator).astype(np.float32)
        return x


# ------------------------------------------------------------------ riff

# (tempo em tempos, duração em tempos, notas, dinâmica 0..1)
AM = ["A2", "E3", "A3", "C4", "E4"]
C = ["C3", "E3", "G3", "C4", "E4"]
G = ["G2", "B2", "D3", "G3", "B3", "G4"]
EM = ["E2", "B2", "E3", "G3", "B3", "E4"]

RIFF = [
    # compasso 1: Lá menor e subida
    (0.0, 2.0, AM, 0.9),
    (2.0, 0.5, ["G3"], 0.7), (2.5, 0.5, ["A3"], 0.6), (3.0, 0.5, ["C4"], 0.75), (3.5, 0.5, ["D4"], 0.65),
    # compasso 2: resposta descendo, termina segurando o Lá
    (4.0, 1.0, ["E4"], 0.85), (5.0, 0.5, ["D4"], 0.6), (5.5, 0.5, ["C4"], 0.65), (6.0, 2.0, ["A3"], 0.8),
    # compasso 3: Dó e Sol abertos
    (8.0, 2.0, C, 0.85), (10.0, 2.0, G, 0.8),
    # compasso 4: Lá menor + frase até o Lá agudo
    (12.0, 1.0, AM, 0.75),
    (13.0, 0.5, ["A3"], 0.6), (13.5, 0.5, ["C4"], 0.65), (14.0, 0.5, ["D4"], 0.7), (14.5, 0.5, ["E4"], 0.75),
    (15.0, 0.5, ["G4"], 0.8), (15.5, 1.5, ["A4"], 0.9),
    # compasso 5: cordas graves, notas curtas
    (17.0, 0.4, ["A2"], 0.9), (17.5, 0.4, ["A2"], 0.6), (18.0, 0.4, ["C3"], 0.85), (18.5, 0.4, ["A2"], 0.6),
    (19.0, 0.4, ["D3"], 0.9), (19.5, 0.4, ["C3"], 0.65), (20.0, 0.9, ["A2"], 0.85), (21.0, 0.9, ["G2"], 0.8),
    # compasso 6: Mi menor de passagem e Lá menor final soando
    (22.0, 1.0, EM, 0.8), (23.0, 3.0, AM, 1.0),
]

STRUM_S = 0.014  # atraso entre cordas no acorde (palhetada para baixo)
RELEASE_S = 0.06  # dedo abafando a corda


def render(bank: Bank) -> np.ndarray:
    total = int((max(t + d for t, d, _, _ in RIFF) * BEAT + 1.0) * SR)
    out = np.zeros(total, np.float32)
    for t, dur, notes, vel in RIFF:
        soft = vel < 0.7
        gain = 0.25 + 0.75 * vel
        for i, n in enumerate(notes):
            x = bank.note(midi(n), soft)
            start = int((t * BEAT + i * STRUM_S) * SR)
            length = min(len(x), int(dur * BEAT * SR) + int(RELEASE_S * SR))
            seg = x[:length].copy()
            rel = int(RELEASE_S * SR)
            if length < len(x) and rel < length:
                seg[-rel:] *= np.linspace(1, 0, rel) ** 2
            g = gain * (0.85 if len(notes) > 1 else 1.0) * (0.92 ** i if len(notes) > 1 else 1.0)
            end = min(start + len(seg), total)
            out[start:end] += g * seg[: end - start]
    # Nível típico de DI numa interface: picos em torno de -6 dBFS.
    out *= 10 ** (-6 / 20) / np.max(np.abs(out))
    fade = int(0.5 * SR)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def main(root: str) -> None:
    bank = Bank(Path(root))
    y = render(bank)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT), y, SR, subtype="PCM_24")
    print(f"{OUT} ({len(y) / SR:.1f} s)")


if __name__ == "__main__":
    main(sys.argv[1])
