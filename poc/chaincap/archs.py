"""Arquiteturas WaveNet "clássicas" do NAM (A1): standard, lite, feather, nano.

A partir do neural-amp-modeler 0.13 o treinador simplificado só treina um
`PackedWaveNet` e exporta um `SlimmableContainer`, que pedaleiras com firmware
antigo não carregam. Aqui recriamos as configurações clássicas no formato de
config da 0.13, para treinar com `nam.train` e exportar um WaveNet comum.

Os tamanhos batem com os do treinador antigo (standard = 13.802 pesos).
"""

from copy import deepcopy as _deepcopy
from typing import Dict, List

_DIL_FULL: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_DIL_LITE_1: List[int] = [1, 2, 4, 8, 16, 32, 64]
_DIL_LITE_2: List[int] = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# (canais da 1ª layer array, canais da 2ª, dilatações 1ª, dilatações 2ª)
_SPECS = {
    "standard": (16, 8, _DIL_FULL, _DIL_FULL),
    "lite": (12, 6, _DIL_LITE_1, _DIL_LITE_2),
    "feather": (8, 4, _DIL_LITE_1, _DIL_LITE_2),
    "nano": (4, 2, _DIL_LITE_1, _DIL_LITE_2),
}

ARCHITECTURES = tuple(_SPECS)


def wavenet_config(arch: str) -> Dict:
    """Config do `nam.models.wavenet.WaveNet` (formato 0.13) para uma arch clássica."""
    if arch not in _SPECS:
        raise ValueError(f"Arquitetura desconhecida {arch!r}; use uma de {ARCHITECTURES}")
    c1, c2, d1, d2 = _SPECS[arch]
    return _deepcopy(
        {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": c1,
                    "head": {"out_channels": c2, "kernel_size": 1, "bias": False},
                    "kernel_size": 3,
                    "dilations": d1,
                    "activation": "Tanh",
                },
                {
                    "input_size": c1,
                    "condition_size": 1,
                    "channels": c2,
                    "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                    "kernel_size": 3,
                    "dilations": d2,
                    "activation": "Tanh",
                },
            ],
            "head": None,
            "head_scale": 0.02,
        }
    )


def lightning_model_config(arch: str) -> Dict:
    """Config completa do LightningModule (rede + loss + otimizador)."""
    return {
        "net": {"name": "WaveNet", "config": wavenet_config(arch)},
        "loss": {"val_loss": "esr", "mrstft_weight": 2e-4},
        "optimizer": {"lr": 0.004},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
    }
