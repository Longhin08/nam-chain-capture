import json
from pathlib import Path

import numpy as np
import torch

import chaincap  # noqa: F401  (shim do tkinter)
from nam.models.wavenet import WaveNet

from chaincap.archs import wavenet_config


def random_wavenet(arch: str, seed: int = 0, scale: float = 0.3) -> WaveNet:
    torch.manual_seed(seed)
    m = WaveNet.init_from_config(wavenet_config(arch))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn_like(p))
    m.sample_rate = 48000
    m.eval()
    return m


def export_nam(model, path: Path, **metadata) -> Path:
    model.export(path.parent, basename=path.stem, other_metadata=metadata or None)
    return path


def write_wav(path: Path, x: np.ndarray, sr: int = 48000) -> Path:
    from scipy.io import wavfile

    wavfile.write(str(path), sr, x.astype(np.float32))
    return path


def write_json(path: Path, d) -> Path:
    path.write_text(json.dumps(d))
    return path
