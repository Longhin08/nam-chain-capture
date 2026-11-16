"""Renderização da cadeia: sinal -> [NAM | IR]* -> saída.

Cada estágio é puro (numpy in, numpy out) para dar para testar isolado.
"""

from . import _compat  # noqa: F401

import json as _json
import logging as _logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
from scipy.io import wavfile as _wavfile
from scipy.signal import fftconvolve as _fftconvolve
from scipy.signal import resample_poly as _resample_poly

from nam.models import init_from_nam as _init_from_nam

from .nam_format import pick_container_submodel, upgrade_model_dict

logger = _logging.getLogger(__name__)

SAMPLE_RATE = 48_000
# Ponto em que o IR "perde a cauda" na captura (ver spec §9).
IR_WARN_MS = 20.0
# Pico máximo permitido no WAV de saída (24-bit clipa em 1.0).
PEAK_LIMIT = 0.98
# Tamanho do bloco para inferência de WaveNets em sinais longos.
_CHUNK = 1 << 19


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass
class Stage:
    type: Literal["nam", "ir"]
    file: Union[str, Path]
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    # Só para IR: truncar a N ms (None = não truncar) e normalizar energia.
    ir_max_ms: Optional[float] = None
    ir_normalize: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any], base_dir: Optional[Path] = None) -> "Stage":
        d = dict(d)
        f = Path(d.pop("file"))
        if base_dir is not None and not f.is_absolute():
            f = base_dir / f
        return cls(file=f, **d)


@dataclass
class ChainResult:
    y: np.ndarray
    warnings: List[str] = field(default_factory=list)
    # Ganho aplicado na saída para não clipar (1.0 = nenhum).
    clip_gain: float = 1.0
    input_level_dbu: Optional[float] = None
    stages_metadata: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------- modelos NAM


def read_nam_file(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as fp:
        d = _json.load(fp)
    for key in ("architecture", "config", "weights"):
        if key not in d:
            raise ValueError(f"{path}: não parece um .nam válido (falta {key!r})")
    if d["architecture"] == "SlimmableContainer":
        d = pick_container_submodel(d)
    return d


def load_nam_model(path: Union[str, Path]):
    """Carrega um `.nam` (0.5.x, 0.7.x ou container) como módulo torch em eval."""
    d = read_nam_file(path)
    sr = d.get("sample_rate")
    if sr is not None and int(sr) != SAMPLE_RATE:
        raise ValueError(
            f"{path}: sample rate {sr} não suportado no PoC (só {SAMPLE_RATE})"
        )
    model = _init_from_nam(upgrade_model_dict(d))
    model.eval()
    return model, d


def nam_infer(model, x: np.ndarray) -> np.ndarray:
    """Roda o modelo no sinal inteiro; saída com o mesmo comprimento (causal)."""
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    rf = getattr(model, "receptive_field", None)
    with torch.no_grad():
        if rf is None or len(x) <= _CHUNK:
            # LSTM/Linear (ou sinal curto): passada única.
            return model(x_t, pad_start=True).numpy().astype(np.float32)
        # WaveNet: blocos com (rf - 1) amostras de contexto -> idêntico à passada única.
        pad = torch.cat([torch.zeros(rf - 1), x_t])
        out = []
        for s in range(0, len(x), _CHUNK):
            e = min(s + _CHUNK, len(x))
            out.append(model(pad[s : e + rf - 1], pad_start=False))
        return torch.cat(out).numpy().astype(np.float32)


# ---------------------------------------------------------------------- IRs


def read_wav_mono(path: Union[str, Path]) -> tuple:
    """Lê WAV (int16/24/32 ou float) e devolve (float32 mono em [-1, 1], sr)."""
    sr, data = _wavfile.read(str(path))
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max + 1)
    data = data.astype(np.float32)
    if data.ndim == 2:
        if data.shape[1] > 1:
            logger.warning("%s: %d canais, usando só o primeiro", path, data.shape[1])
        data = data[:, 0]
    return data, int(sr)


def load_ir(
    path: Union[str, Path],
    max_ms: Optional[float] = None,
    normalize: bool = True,
) -> tuple:
    """Carrega um IR em 48 kHz. Retorna (ir, avisos)."""
    warnings: List[str] = []
    h, sr = read_wav_mono(path)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        h = _resample_poly(h, SAMPLE_RATE // g, sr // g).astype(np.float32)
        warnings.append(f"IR {Path(path).name}: reamostrado de {sr} para {SAMPLE_RATE} Hz")
    if max_ms is not None:
        n = int(round(max_ms * SAMPLE_RATE / 1000))
        if len(h) > n:
            h = h[:n].copy()
            warnings.append(f"IR {Path(path).name}: truncado em {max_ms:g} ms")
    length_ms = 1000.0 * len(h) / SAMPLE_RATE
    if length_ms > IR_WARN_MS:
        warnings.append(
            f"IR {Path(path).name} tem {length_ms:.0f} ms (> {IR_WARN_MS:g} ms): "
            "a cauda pode não ser capturada com fidelidade pelo modelo final"
        )
    if normalize:
        energy = float(np.sqrt(np.sum(h.astype(np.float64) ** 2)))
        if energy == 0.0:
            raise ValueError(f"IR {path} é todo zero")
        h = (h / energy).astype(np.float32)
    return h, warnings


def apply_ir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolução causal, mesmo comprimento do input."""
    return _fftconvolve(x, h, mode="full")[: len(x)].astype(np.float32)


# ------------------------------------------------------------------- cadeia


def render_chain(x: np.ndarray, stages: Sequence[Stage]) -> ChainResult:
    """Passa `x` por todos os estágios, na ordem dada (nunca reordena)."""
    if not stages:
        raise ValueError("A cadeia precisa de pelo menos um estágio")
    warnings: List[str] = []
    meta: List[Dict[str, Any]] = []
    input_level_dbu: Optional[float] = None
    y = np.asarray(x, dtype=np.float32)
    first_nam_seen = False
    pre_gain_db = 0.0  # ganho acumulado antes do 1º NAM (afeta a calibração)

    for i, st in enumerate(stages):
        y = y * db_to_gain(st.input_gain_db)
        entry: Dict[str, Any] = {
            "type": st.type,
            "file": Path(st.file).name,
            "input_gain_db": st.input_gain_db,
            "output_gain_db": st.output_gain_db,
        }
        if st.type == "nam":
            model, d = load_nam_model(st.file)
            md = d.get("metadata") or {}
            entry.update(
                architecture=d.get("architecture"),
                version=d.get("version"),
                name=md.get("name"),
                modeled_by=md.get("modeled_by"),
            )
            if not first_nam_seen:
                first_nam_seen = True
                lvl = md.get("input_level_dbu")
                if lvl is not None:
                    # Se o sinal entra +g dB no modelo, a calibração efetiva do
                    # conjunto cai g dB.
                    input_level_dbu = float(lvl) - (pre_gain_db + st.input_gain_db)
            y = nam_infer(model, y)
        elif st.type == "ir":
            h, w = load_ir(st.file, max_ms=st.ir_max_ms, normalize=st.ir_normalize)
            warnings.extend(w)
            entry["ir_samples"] = int(len(h))
            y = apply_ir(y, h)
        else:
            raise ValueError(f"Estágio {i}: tipo desconhecido {st.type!r}")
        y = y * db_to_gain(st.output_gain_db)
        if not first_nam_seen:
            pre_gain_db += st.input_gain_db + st.output_gain_db
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"Estágio {i} ({entry['file']}) gerou NaN/inf")
        meta.append(entry)

    clip_gain = 1.0
    peak = float(np.max(np.abs(y)))
    if peak >= PEAK_LIMIT:
        clip_gain = PEAK_LIMIT / peak
        y = y * clip_gain
        warnings.append(
            f"Saída da cadeia clipava (pico {peak:.2f}); aplicado "
            f"{20 * np.log10(clip_gain):.1f} dB no último estágio"
        )
    if peak == 0.0:
        raise ValueError("A cadeia produziu silêncio")

    for w in warnings:
        logger.warning(w)
    return ChainResult(
        y=y.astype(np.float32),
        warnings=warnings,
        clip_gain=clip_gain,
        input_level_dbu=input_level_dbu,
        stages_metadata=meta,
    )


def write_wav_24(path: Union[str, Path], y: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Grava WAV PCM 24-bit mono, sem fade nem normalização."""
    from nam.data import np_to_wav

    np_to_wav(y, path, rate=sr, sampwidth=3)
