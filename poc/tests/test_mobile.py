"""App de celular (mobile/www): a prévia soa como o motor Python e o pacote abre no Colab."""

import copy
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from chaincap import colab
from chaincap.cab_ir import write_cab_ir
from chaincap.chain import parse_node, render_chain

from conftest import FIXTURES

RUNNER = Path(__file__).resolve().parents[2] / "mobile" / "tests-run.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="sem Node")


def _node(tmp_path, mode, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    subprocess.run(["node", str(RUNNER), mode, str(path)], check=True)


def _a2_nam(path, channels=3, seed=1):
    """`.nam` com a estrutura A2 (container, LeakyReLU, kernels mistos) e pesos aleatórios."""
    kernels = [6, 6, 15, 6] * 3
    dilations = [1, 3, 7, 17, 41, 101, 1, 3, 7, 17, 41, 101]
    off = {"active": False, "shift": True, "groups": 1}
    layer = {
        "input_size": 1, "condition_size": 1, "channels": channels, "bottleneck": channels,
        "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
        "kernel_sizes": kernels, "dilations": dilations,
        "activation": [{"type": "LeakyReLU", "negative_slope": 0.01}] * len(kernels),
        "gating_mode": ["none"] * len(kernels), "secondary_activation": [None] * len(kernels),
        "head1x1": {"active": False, "out_channels": 1, "groups": 1},
        "layer1x1": {"active": True, "groups": 1}, "groups_input": 1, "groups_input_mixin": 1,
        "conv_pre_film": off, "conv_post_film": off, "input_mixin_pre_film": off,
        "input_mixin_post_film": off, "activation_pre_film": off, "activation_post_film": off,
        "layer1x1_post_film": off, "head1x1_post_film": off, "slimmable": None,
    }
    c = channels
    count = c + sum(c * c * k + c + c + c * c + c for k in kernels) + c * 16 + 1
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal(count) * 0.15).tolist() + [0.05]
    model = {"version": "0.7.0", "architecture": "WaveNet", "sample_rate": 48000,
             "config": {"layers": [layer], "head": None, "head_scale": 0.05}, "weights": weights}
    path.write_text(json.dumps({
        "version": "0.7.0", "architecture": "SlimmableContainer", "sample_rate": 48000,
        "config": {"submodels": [{"max_value": 1.0, "model": model}]}, "weights": []}))
    return path


def _st(path, kind="nam", i=0.0, o=0.0, **extra):
    return {"type": kind, "file": str(path), "input_gain_db": i, "output_gain_db": o, **extra}


@pytest.fixture
def chains(tmp_path):
    a2 = _a2_nam(tmp_path / "a2.nam")
    cab = tmp_path / "cab.wav"
    write_cab_ir(cab)
    ir = _st(cab, "ir", o=-6.0, ir_max_ms=20)
    nano, gated = FIXTURES / "legacy_nano.nam", FIXTURES / "legacy_gated.nam"
    split = {"type": "split", "branches": [
        {"level_db": -7.5, "invert": True, "enabled": True, "stages": [_st(gated, i=-6)]},
        {"level_db": -3.0, "invert": False, "enabled": True, "stages": [_st(a2), ir]},
        {"level_db": -9.0, "invert": False, "enabled": True, "stages": []},  # sinal limpo
        {"level_db": 0.0, "invert": False, "enabled": False, "stages": [ir]},  # no mudo
    ]}
    return {"a1": [_st(nano, i=3, o=-2)], "a1_gated": [_st(gated)], "a2": [_st(a2, i=-4)],
            "ir": [ir], "split": [_st(nano, i=-6), split],
            "clipa": [_st(nano, o=30)]}


@pytest.mark.parametrize("simd", [True, False], ids=["simd", "sem-simd"])
@pytest.mark.parametrize("name", ["a1", "a1_gated", "a2", "ir", "split", "clipa"])
def test_preview_matches_the_python_engine(tmp_path, chains, name, simd):
    # mais longo que um bloco do app, para conferir a emenda entre blocos
    x = (np.random.default_rng(0).standard_normal(40_000) * 0.1).astype(np.float32)
    ref = render_chain(x, [parse_node(copy.deepcopy(n)) for n in chains[name]]).y
    x.tofile(tmp_path / "in.f32")
    _node(tmp_path, "render", {"nodes": chains[name], "input": str(tmp_path / "in.f32"),
                               "output": str(tmp_path / "out.f32"), "noSimd": not simd})
    got = np.fromfile(tmp_path / "out.f32", np.float32)
    assert got.shape == ref.shape
    esr = np.sum((got - ref).astype(np.float64) ** 2) / np.sum(ref.astype(np.float64) ** 2)
    assert esr < 1e-9


def test_phone_package_opens_in_colab(tmp_path, chains):
    """O zip montado no celular passa pelo mesmo código que o notebook usa."""
    nodes = copy.deepcopy(chains["split"])
    nodes[0]["fileName"] = "../Ampli ção.NAM"
    nodes[1]["branches"][0]["stages"][0]["fileName"] = "..\\Ampli ção.nam"
    out = tmp_path / "celular.chaincap"
    _node(tmp_path, "package", {"nodes": nodes, "name": "Timbre da Márcia", "output": str(out)})
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None  # CRCs conferem
    job = colab.open_package(out, tmp_path / "w")
    assert job["name"] == "Timbre da Márcia" and job["arch"] == "a2" and job["epochs"] == 100
    chain = json.loads(Path(job["chain"]).read_text(encoding="utf-8"))
    files = colab._chain_files(chain)
    assert all(f.count("/") == 1 and "\\" not in f for f in files)
    assert len({f.lower() for f in files[:2]}) == 2
    # e o motor Python renderiza a cadeia extraída
    base = Path(job["dir"])
    parsed = [parse_node(n, base) for n in chain["stages"]]
    x = (np.random.default_rng(0).standard_normal(4800) * 0.1).astype(np.float32)
    assert np.any(render_chain(x, parsed).y)
