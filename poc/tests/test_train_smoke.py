"""Smoke test do pipeline completo (render -> treino de 1 passo -> export)."""

import json

import numpy as np

from chaincap.__main__ import main
from chaincap.chain import write_wav_24

from conftest import FIXTURES


def test_cli_run_smoke(tmp_path):
    # Sinal sintético com o mesmo comprimento do v3 não é necessário: basta caber
    # nos splits do v3 (treino começa em 480k, validação são os últimos 432k).
    rng = np.random.default_rng(0)
    n = 1_200_000
    x = (rng.standard_normal(n) * 0.1).astype(np.float32)
    x[480_000 - 48_000 : 480_000] = 0  # silêncio antes do treino
    x[-432_000 - 48_000 : -432_000] = 0  # e antes da validação
    inp = tmp_path / "input.wav"
    write_wav_24(inp, x)

    ir = np.zeros(32, np.float32)
    ir[:4] = [1.0, 0.5, 0.25, 0.1]
    from helpers import write_wav

    write_wav(tmp_path / "cab.wav", ir)
    chain = {
        "name": "smoke",
        "arch": "nano",
        "stages": [
            {"type": "nam", "file": str(FIXTURES / "legacy_nano.nam"), "input_gain_db": 3},
            {"type": "ir", "file": "cab.wav"},
        ],
    }
    (tmp_path / "chain.json").write_text(json.dumps(chain))

    out = tmp_path / "out"
    rc = main([
        "run", str(tmp_path / "chain.json"), "--input", str(inp), "--out", str(out),
        "--smoke", "--epochs", "1", "--skip-input-check", "--gpu-usd-per-hour", "0.5",
    ])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["arch"] == "nano"
    assert np.isfinite(report["esr"])

    nam = json.loads((out / "chain.nam").read_text())
    md = nam["metadata"]
    assert md["name"] == "smoke"
    assert md["input_level_dbu"] == 9.0
    assert [s["type"] for s in md["chain_capture"]["stages"]] == ["nam", "ir"]
    assert md["chain_capture"]["arch"] == "nano"

    a1 = json.loads((out / "chain.a1.nam").read_text())
    assert a1["version"] == "0.5.4"
    assert a1["weights"] == nam["weights"]


def test_cli_run_smoke_with_split(tmp_path):
    """O worker de treino entende a divisão no chain.json."""
    rng = np.random.default_rng(1)
    n = 1_200_000
    x = (rng.standard_normal(n) * 0.1).astype(np.float32)
    x[480_000 - 48_000 : 480_000] = 0
    x[-432_000 - 48_000 : -432_000] = 0
    inp = tmp_path / "input.wav"
    write_wav_24(inp, x)
    chain = {
        "name": "dual",
        "arch": "nano",
        "stages": [
            {"type": "split", "branches": [
                {"stages": [{"type": "nam", "file": str(FIXTURES / "legacy_nano.nam")}], "level_db": -3},
                {"stages": [{"type": "nam", "file": str(FIXTURES / "legacy_gated.nam")}], "invert": True},
            ]},
        ],
    }
    (tmp_path / "chain.json").write_text(json.dumps(chain))
    out = tmp_path / "out"
    rc = main(["run", str(tmp_path / "chain.json"), "--input", str(inp), "--out", str(out),
               "--smoke", "--epochs", "1", "--skip-input-check"])
    assert rc == 0
    md = json.loads((out / "chain.nam").read_text())["metadata"]["chain_capture"]
    split = md["stages"][0]
    assert split["type"] == "split" and len(split["branches"]) == 2
    assert split["branches"][1]["invert"] is True
    assert (out / "chain.a1.nam").is_file()


def _synthetic_input(tmp_path, seed=2):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(1_200_000) * 0.1).astype(np.float32)
    x[480_000 - 48_000 : 480_000] = 0
    x[-432_000 - 48_000 : -432_000] = 0
    inp = tmp_path / "input.wav"
    write_wav_24(inp, x)
    return inp


def test_cli_run_smoke_a2_matches_tone3000_format(tmp_path):
    """A2: mesmo formato das capturas atuais do Tone3000 (aceito pela M-VAVE Pocket Amp)."""
    from chaincap.chain import load_nam_model, nam_infer

    inp = _synthetic_input(tmp_path)
    chain = {"name": "Meu A2", "arch": "a2",
             "stages": [{"type": "nam", "file": str(FIXTURES / "legacy_nano.nam")}]}
    (tmp_path / "chain.json").write_text(json.dumps(chain))
    out = tmp_path / "out"
    rc = main(["run", str(tmp_path / "chain.json"), "--input", str(inp), "--out", str(out),
               "--smoke", "--epochs", "1", "--skip-input-check"])
    assert rc == 0
    d = json.loads((out / "chain.nam").read_text())
    assert d["version"] == "0.7.0" and d["architecture"] == "SlimmableContainer"
    assert d["sample_rate"] == 48000
    subs = d["config"]["submodels"]
    assert [s["max_value"] for s in subs] == [0.5, 1]
    assert [s["model"]["config"]["layers"][0]["channels"] for s in subs] == [3, 8]
    assert [len(s["model"]["weights"]) for s in subs] == [1871, 12146]  # = Tone3000
    assert not (out / "chain.a1.nam").exists()  # A2 não tem versão A1

    md = d["metadata"]
    assert md["name"] == "Meu A2" and md["gear_type"] == "amp"  # captura de teste sem gear_type
    def has_null(v):
        if v is None:
            return True
        if isinstance(v, dict):
            return any(has_null(x) for x in v.values())
        if isinstance(v, list):
            return any(has_null(x) for x in v)
        return False

    assert not has_null(md)  # metadados sem null (a config do modelo é a padrão do NAM)

    report = json.loads((out / "report.json").read_text())
    assert report["arch"] == "a2" and len(report["esr_by_size"]) == 2

    model, _ = load_nam_model(out / "chain.nam")  # o app carrega o próprio resultado
    y = nam_infer(model, np.zeros(4800, np.float32) + 0.01)
    assert np.all(np.isfinite(y))
