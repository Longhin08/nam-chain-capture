"""Atalhos de tempo de treino: warm start, parada automática e cache do render."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chaincap.__main__ import main
from chaincap.archs import lightning_model_config
from chaincap.chain import load_nam_model, nam_infer, parse_node, warm_start_candidates
from chaincap.train import AutoStop, warm_start
from nam.train.lightning_module import LightningModule, PackedLightningModule

from conftest import FIXTURES
from test_train_smoke import _synthetic_input


@pytest.fixture(scope="module")
def a2_capture(tmp_path_factory):
    """Uma captura A2 (container, como as do Tone3000) gerada pelo próprio app."""
    tmp = tmp_path_factory.mktemp("a2")
    inp = _synthetic_input(tmp)
    chain = {"name": "Amp A2", "arch": "a2",
             "stages": [{"type": "nam", "file": str(FIXTURES / "legacy_nano.nam")}]}
    (tmp / "chain.json").write_text(json.dumps(chain))
    rc = main(["run", str(tmp / "chain.json"), "--input", str(inp), "--out", str(tmp / "out"),
               "--smoke", "--epochs", "1", "--skip-input-check"])
    assert rc == 0
    return tmp / "out" / "chain.nam"


def _packed():
    m = PackedLightningModule.init_from_config(lightning_model_config("a2"))
    m.net.sample_rate = 48000
    return m


def _signal(n=24000, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


# ------------------------------------------------------------------ warm start


def test_warm_start_reproduces_capture_even_with_other_head_scale(a2_capture, tmp_path):
    # Capturas do Tone3000 trazem head_scale aprendido (≠ 0.01 do treinador).
    d = json.loads(a2_capture.read_text())
    for s in d["config"]["submodels"]:
        s["model"]["config"]["head_scale"] = 0.0123
    f = tmp_path / "amp.nam"
    f.write_text(json.dumps(d))

    m = _packed()
    assert warm_start(m, packed=True, nam_path=f)
    ref, _ = load_nam_model(f)  # o app usa o submodelo maior
    big = m.net.extract_submodel(m.net.num_submodels - 1).eval()
    x = _signal()
    np.testing.assert_allclose(nam_infer(big, x), nam_infer(ref, x), atol=1e-5)


def test_warm_start_refuses_other_structures(a2_capture):
    # A1 nano não encaixa no A2; A2 não encaixa no A1 standard.
    assert not warm_start(_packed(), packed=True, nam_path=FIXTURES / "legacy_nano.nam")
    std = LightningModule.init_from_config(lightning_model_config("standard"))
    assert not warm_start(std, packed=False, nam_path=a2_capture)


def test_warm_start_classic_same_arch():
    m = LightningModule.init_from_config(lightning_model_config("nano"))
    if not warm_start(m, packed=False, nam_path=FIXTURES / "legacy_nano.nam"):
        pytest.skip("fixture nano tem outra estrutura que o nano do app")
    ref, _ = load_nam_model(FIXTURES / "legacy_nano.nam")
    x = _signal()
    np.testing.assert_allclose(nam_infer(m.net.eval(), x), nam_infer(ref, x), atol=1e-5)


def _nam_with_gear(tmp_path, name, gear):
    d = json.loads((FIXTURES / "legacy_nano.nam").read_text())
    d.setdefault("metadata", {})["gear_type"] = gear
    p = tmp_path / f"{name}.nam"
    p.write_text(json.dumps(d))
    return p


def test_warm_start_candidates_prefers_amp_and_loudest_branch(tmp_path):
    pedal = _nam_with_gear(tmp_path, "pedal", "pedal")
    amp = _nam_with_gear(tmp_path, "amp", "amp")
    amp_b = _nam_with_gear(tmp_path, "ampB", "amp_cab")
    nodes = [parse_node(n) for n in (
        {"type": "nam", "file": str(pedal)},
        {"type": "split", "branches": [
            {"stages": [{"type": "nam", "file": str(amp)}], "level_db": -6},
            {"stages": [{"type": "nam", "file": str(amp_b)}], "level_db": 0},
        ]},
    )]
    assert warm_start_candidates(nodes) == [amp_b, pedal]
    only = [parse_node({"type": "nam", "file": str(pedal)}), parse_node({"type": "nam", "file": str(amp)})]
    assert warm_start_candidates(only) == [amp, pedal]


def test_cli_run_reports_warm_start(a2_capture, tmp_path):
    inp = _synthetic_input(tmp_path, seed=5)
    chain = {"name": "Com warm", "arch": "a2", "stages": [{"type": "nam", "file": str(a2_capture)}]}
    (tmp_path / "chain.json").write_text(json.dumps(chain))
    out = tmp_path / "out"
    assert main(["run", str(tmp_path / "chain.json"), "--input", str(inp), "--out", str(out),
                 "--smoke", "--epochs", "1", "--skip-input-check"]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["warm_start"] == "chain.nam" and rep["epochs_max"] == 1
    md = json.loads((out / "chain.nam").read_text())["metadata"]
    assert md["chain_capture"]["warm_start"] == "chain.nam"

    out2 = tmp_path / "out2"
    assert main(["run", str(tmp_path / "chain.json"), "--input", str(inp), "--out", str(out2),
                 "--smoke", "--epochs", "1", "--skip-input-check", "--no-warm-start"]) == 0
    assert json.loads((out2 / "report.json").read_text())["warm_start"] is None


# ------------------------------------------------------------- parada automática


def _feed(stop, esrs, packed=True):
    tr = SimpleNamespace(sanity_checking=False, should_stop=False, current_epoch=0, callback_metrics={})
    for i, e in enumerate(esrs):
        tr.current_epoch = i
        tr.callback_metrics = ({"ESR_packed_0": torch.tensor(e * 3), "ESR_packed_1": torch.tensor(e)}
                               if packed else {"ESR": torch.tensor(e)})
        stop.on_validation_epoch_end(tr, None)
        if tr.should_stop:
            return i + 1
    return None


def test_auto_stop_on_target_uses_the_pedal_model_esr():
    # Para pelo ESR do modelo maior (o do pedal), não pela soma dos dois tamanhos.
    s = AutoStop(target_esr=0.002, min_epochs=3)
    assert _feed(s, [0.05, 0.01, 0.0015, 0.001]) == 3
    assert "alvo" in s.reason


def test_auto_stop_on_plateau_but_not_before_min_epochs():
    s = AutoStop(target_esr=None, patience=3, rel_delta=0.03, min_epochs=6)
    esrs = [0.1, 0.05, 0.04, 0.0399, 0.0398, 0.0399, 0.0397, 0.0398]
    assert _feed(s, esrs, packed=False) == 6  # platô desde a época 3, mas espera o mínimo
    assert "parou de melhorar" in s.reason
    s2 = AutoStop(target_esr=None, patience=3, min_epochs=1)
    assert _feed(s2, [0.1, 0.08, 0.06, 0.05, 0.04]) is None  # ainda melhorando


# ---------------------------------------------------------------- cache do render


def test_render_cache_reuses_same_chain_only(tmp_path, monkeypatch):
    inp = _synthetic_input(tmp_path, seed=7)
    chain = {"name": "c", "stages": [{"type": "nam", "file": str(FIXTURES / "legacy_nano.nam")}]}
    (tmp_path / "chain.json").write_text(json.dumps(chain))
    cache = tmp_path / "cache"
    args = lambda out: ["render", str(tmp_path / "chain.json"), "--input", str(inp),
                        "--out", str(tmp_path / out), "--cache-dir", str(cache)]
    assert main(args("o1")) == 0
    assert len(list(cache.glob("*.wav"))) == 1

    import chaincap.__main__ as cli

    def boom(*a, **k):
        raise AssertionError("não devia renderizar de novo")

    monkeypatch.setattr(cli, "render_chain", boom)
    assert main(args("o2")) == 0  # mesma cadeia: vem do cache
    assert (tmp_path / "o1/target.wav").read_bytes() == (tmp_path / "o2/target.wav").read_bytes()

    monkeypatch.undo()
    chain["stages"][0]["input_gain_db"] = 3  # mudou a cadeia: render novo
    (tmp_path / "chain.json").write_text(json.dumps(chain))
    assert main(args("o3")) == 0
    assert len(list(cache.glob("*.wav"))) == 2
