"""Conversão entre o formato `.nam` legado (0.5.x, "A1") e o atual (0.7.x).

- Capturas antigas (a maioria no Tone3000) usam `head_size`/`head_bias`/`gated`
  e não carregam direto no `init_from_nam` da 0.13 -> `upgrade_wavenet_config`.
- Pedaleiras com firmware antigo só leem o formato 0.5.x -> `to_legacy_a1`.
- `SlimmableContainer` (saída padrão do treinador 0.13) -> escolhemos o
  submodelo de maior qualidade com `pick_container_submodel`.

A ordem dos pesos é a mesma nos dois formatos para WaveNets clássicos; só o
JSON da config muda.
"""

from copy import deepcopy as _deepcopy
from typing import Any, Dict

LEGACY_VERSION = "0.5.4"

_FILM_KEYS = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


class UnsupportedForLegacyExport(ValueError):
    """O modelo usa recursos que não existem no formato 0.5.x."""


def is_legacy_layer(layer: Dict[str, Any]) -> bool:
    return "head_size" in layer and "head" not in layer


def upgrade_wavenet_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Converte a `config` de um WaveNet 0.5.x para o formato que a 0.13 entende."""
    config = _deepcopy(config)
    for layer in config["layers"]:
        if not is_legacy_layer(layer):
            continue
        layer["head"] = {
            "out_channels": layer.pop("head_size"),
            "kernel_size": 1,
            "bias": bool(layer.pop("head_bias")),
        }
        gated = bool(layer.pop("gated", False))
        act = layer["activation"]
        if gated:
            layer["activation"] = {
                "name": "PairMultiply",
                "primary": act,
                "secondary": "Sigmoid",
            }
    return config


def upgrade_model_dict(model: Dict[str, Any]) -> Dict[str, Any]:
    """Devolve uma cópia do dict `.nam` pronta para `nam.models.init_from_nam`."""
    model = _deepcopy(model)
    if model.get("architecture") == "WaveNet":
        model["config"] = upgrade_wavenet_config(model["config"])
    return model


def pick_container_submodel(container: Dict[str, Any]) -> Dict[str, Any]:
    """Escolhe o submodelo de maior `max_value` (maior qualidade) de um container."""
    subs = container["config"]["submodels"]
    best = max(subs, key=lambda s: s.get("max_value", 0.0))["model"]
    best = _deepcopy(best)
    if "sample_rate" not in best and "sample_rate" in container:
        best["sample_rate"] = container["sample_rate"]
    meta = dict(container.get("metadata") or {})
    meta.update(best.get("metadata") or {})
    best["metadata"] = meta
    return best


def _all_equal(xs) -> bool:
    return len(set(map(lambda v: repr(v), xs))) <= 1


def to_legacy_a1(model: Dict[str, Any]) -> Dict[str, Any]:
    """Converte um `.nam` WaveNet 0.7.x para o formato 0.5.x (A1).

    Levanta `UnsupportedForLegacyExport` se o modelo usar algo que o formato
    antigo não representa (FiLM, grupos, bottleneck, kernels mistos, etc.).
    """
    if model.get("architecture") != "WaveNet":
        raise UnsupportedForLegacyExport(
            f"Só WaveNet é suportado (recebido {model.get('architecture')!r})"
        )
    out = _deepcopy(model)
    cfg = out["config"]
    legacy_layers = []
    for i, layer in enumerate(cfg["layers"]):
        if is_legacy_layer(layer):
            legacy_layers.append(layer)
            continue

        def fail(why: str):
            raise UnsupportedForLegacyExport(f"layer array {i}: {why}")

        head = layer["head"]
        if head.get("kernel_size", 1) != 1:
            fail("head com kernel_size != 1")
        if layer.get("bottleneck", layer["channels"]) != layer["channels"]:
            fail("bottleneck != channels")
        if layer.get("head1x1", {}).get("active", False):
            fail("head1x1 ativo")
        if not layer.get("layer1x1", {}).get("active", True):
            fail("layer1x1 desativado")
        if layer.get("groups_input", 1) != 1 or layer.get("groups_input_mixin", 1) != 1:
            fail("convoluções agrupadas")
        for k in _FILM_KEYS:
            if layer.get(k, {}).get("active", False):
                fail(f"FiLM {k} ativo")
        if layer.get("slimmable"):
            fail("camada slimmable")

        ks = layer.get("kernel_sizes", layer.get("kernel_size"))
        if isinstance(ks, list):
            if not _all_equal(ks):
                fail("kernel sizes diferentes entre camadas")
            ks = ks[0]

        acts = layer["activation"]
        if not isinstance(acts, list):
            acts = [acts]
        gmodes = layer.get("gating_mode", ["none"] * len(acts))
        secs = layer.get("secondary_activation", [None] * len(acts))
        if not (_all_equal(acts) and _all_equal(gmodes) and _all_equal(secs)):
            fail("ativações diferentes entre camadas")
        act = acts[0]
        if isinstance(act, dict):
            if set(act) != {"type"}:
                fail(f"ativação com parâmetros {act}")
            act = act["type"]
        gmode = gmodes[0]
        if gmode == "none":
            gated = False
        elif gmode == "gated" and (secs[0] or {}).get("type") == "Sigmoid":
            gated = True
        else:
            fail(f"gating {gmode!r} não suportado")

        legacy_layers.append(
            {
                "input_size": layer["input_size"],
                "condition_size": layer["condition_size"],
                "head_size": head["out_channels"],
                "channels": layer["channels"],
                "kernel_size": ks,
                "dilations": layer["dilations"],
                "activation": act,
                "gated": gated,
                "head_bias": bool(head["bias"]),
            }
        )
    if cfg.get("head") is not None:
        raise UnsupportedForLegacyExport("head pós-layers não suportado")
    out["config"] = {
        "layers": legacy_layers,
        "head": None,
        "head_scale": cfg["head_scale"],
    }
    out["version"] = LEGACY_VERSION
    return out
