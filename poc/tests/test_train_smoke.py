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
