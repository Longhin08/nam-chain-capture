import json

import numpy as np
import pytest
import torch

from chaincap.archs import ARCHITECTURES
from chaincap.chain import load_nam_model, nam_infer
from chaincap.nam_format import (
    UnsupportedForLegacyExport,
    pick_container_submodel,
    to_legacy_a1,
    upgrade_model_dict,
)
from nam.models import init_from_nam

from conftest import FIXTURES
from helpers import export_nam, random_wavenet

REFS = np.load(FIXTURES / "legacy_refs.npz")

# Números de pesos das archs clássicas no treinador antigo (0.10).
EXPECTED_WEIGHTS = {"standard": 13802, "lite": 6554, "feather": 3026, "nano": 842}


@pytest.mark.parametrize("name", ["legacy_nano", "legacy_gated", "legacy_lstm"])
def test_legacy_files_load_and_match_old_trainer(name):
    """Arquivos 0.5.4 gerados pelo NAM 0.10 rodam igual no 0.13 após upgrade."""
    model, d = load_nam_model(FIXTURES / f"{name}.nam")
    assert d["version"] == "0.5.4"
    y = nam_infer(model, REFS["x"])
    # O LSTM do NAM faz um "burn-in" do estado inicial que mudou entre versões;
    # depois do transiente inicial a saída tem que ser idêntica.
    start = 2000 if name == "legacy_lstm" else 0
    np.testing.assert_allclose(y[start:], REFS[name][start:], atol=1e-5)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_arch_weight_counts_match_classic(arch, tmp_path):
    m = random_wavenet(arch)
    d = json.loads(export_nam(m, tmp_path / "m.nam").read_text())
    assert len(d["weights"]) == EXPECTED_WEIGHTS[arch]


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_legacy_export_roundtrip(arch, tmp_path):
    m = random_wavenet(arch, seed=3)
    d = json.loads(export_nam(m, tmp_path / "m.nam", name="x").read_text())
    legacy = to_legacy_a1(d)
    assert legacy["version"] == "0.5.4"
    for layer in legacy["config"]["layers"]:
        assert set(layer) == {
            "input_size", "condition_size", "head_size", "channels",
            "kernel_size", "dilations", "activation", "gated", "head_bias",
        }
        assert layer["activation"] == "Tanh" and layer["gated"] is False
    assert legacy["weights"] == d["weights"]
    assert legacy["metadata"]["name"] == "x"

    m2 = init_from_nam(upgrade_model_dict(legacy)).eval()
    x = torch.randn(20000) * 0.3
    with torch.no_grad():
        torch.testing.assert_close(m(x, pad_start=True), m2(x, pad_start=True))


def test_legacy_export_is_idempotent_on_legacy_input():
    d = json.loads((FIXTURES / "legacy_nano.nam").read_text())
    assert to_legacy_a1(d)["config"]["layers"] == d["config"]["layers"]


def test_legacy_export_rejects_non_wavenet():
    d = json.loads((FIXTURES / "legacy_lstm.nam").read_text())
    with pytest.raises(UnsupportedForLegacyExport):
        to_legacy_a1(d)


def test_legacy_export_rejects_film(tmp_path):
    m = random_wavenet("nano")
    d = json.loads(export_nam(m, tmp_path / "m.nam").read_text())
    d["config"]["layers"][0]["conv_pre_film"]["active"] = True
    with pytest.raises(UnsupportedForLegacyExport, match="FiLM"):
        to_legacy_a1(d)


def test_container_picks_highest_quality_submodel(tmp_path):
    small = json.loads(export_nam(random_wavenet("nano"), tmp_path / "a.nam").read_text())
    big = json.loads(export_nam(random_wavenet("feather"), tmp_path / "b.nam").read_text())
    container = {
        "version": "0.7.0",
        "architecture": "SlimmableContainer",
        "metadata": {"name": "rig"},
        "sample_rate": 48000,
        "config": {"submodels": [
            {"max_value": 0.5, "model": small},
            {"max_value": 1.0, "model": big},
        ]},
        "weights": [],
    }
    picked = pick_container_submodel(container)
    assert len(picked["weights"]) == EXPECTED_WEIGHTS["feather"]
    assert picked["metadata"]["name"] == "rig"

    (tmp_path / "c.nam").write_text(json.dumps(container))
    model, _ = load_nam_model(tmp_path / "c.nam")
    # o .nam guarda head_scale como 1 peso extra além dos parâmetros
    assert sum(p.numel() for p in model.parameters()) + 1 == EXPECTED_WEIGHTS["feather"]
