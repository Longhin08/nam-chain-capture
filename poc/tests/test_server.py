"""Testes da API da interface (sem rodar treino de verdade)."""

import io
import json
import os
import tempfile
import threading

import numpy as np
import pytest
from scipy.io import wavfile

from conftest import FIXTURES

_WS = tempfile.mkdtemp(prefix="chaincap-ws-")
os.environ["CHAINCAP_WORKSPACE"] = _WS

from fastapi.testclient import TestClient  # noqa: E402

from app import server  # noqa: E402

client = TestClient(server.app)


def wav_bytes(x, sr=48000):
    buf = io.BytesIO()
    wavfile.write(buf, sr, x.astype(np.float32))
    return buf.getvalue()


def upload(kind, name, data):
    return client.post("/api/files", data={"kind": kind}, files={"file": (name, data)})


@pytest.fixture(scope="module", autouse=True)
def library():
    nam = (FIXTURES / "legacy_nano.nam").read_bytes()
    assert upload("nam", "drive.nam", nam).status_code == 200
    ir = np.zeros(64, np.float32)
    ir[:3] = [1, 0.5, 0.2]
    assert upload("ir", "cab.wav", wav_bytes(ir)).status_code == 200
    di = (np.random.default_rng(0).standard_normal(48000 * 3) * 0.1).astype(np.float32)
    assert upload("di", "riff.wav", wav_bytes(di)).status_code == 200


def test_list_files_describes_each_kind():
    d = client.get("/api/files").json()
    assert [f["name"] for f in d["nam"]] == ["drive.nam"]
    assert d["nam"][0]["architecture"] == "WaveNet"
    assert d["nam"][0]["input_level_dbu"] == 12.0
    assert d["ir"][0]["seconds"] == pytest.approx(64 / 48000, abs=1e-3)
    assert d["di"][0]["seconds"] == pytest.approx(3.0)


def test_upload_rejects_wrong_extension_and_invalid_content():
    assert upload("nam", "x.wav", b"abc").status_code == 400
    r = upload("nam", "bad.nam", b'{"foo": 1}')
    assert r.status_code == 400 and "válido" in r.json()["detail"]
    assert not any(f["name"] == "bad.nam" for f in client.get("/api/files").json()["nam"])


def test_upload_does_not_overwrite_and_sanitizes_names():
    nam = (FIXTURES / "legacy_nano.nam").read_bytes()
    r = upload("nam", "../../evil/drive.nam", nam)
    assert r.status_code == 200
    assert r.json()["name"] == "drive (2).nam"
    assert client.delete("/api/files/nam/drive (2).nam").status_code == 200


def test_delete_cannot_escape_directory():
    for path in ("/api/files/nam/..%2F..%2Fjobs", "/api/files/nam/..", "/api/files/nam/%2E%2E"):
        assert 400 <= client.delete(path).status_code < 500
    assert server.JOBS.is_dir() and (server.FILES / "nam" / "drive.nam").is_file()
    assert client.delete("/api/files/zzz/drive.nam").status_code == 400


def test_preview_returns_audio_with_level_headers():
    r = client.post("/api/preview", json={
        "di": "riff.wav", "seconds": 2,
        "stages": [
            {"type": "nam", "file": "drive.nam", "input_gain_db": 3},
            {"type": "ir", "file": "cab.wav", "ir_max_ms": 20},
        ],
    })
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    sr, y = wavfile.read(io.BytesIO(r.content))
    assert sr == 48000 and len(y) == 96000
    assert float(r.headers["X-Chain-Peak-Dbfs"]) < 10
    json.loads(r.headers["X-Chain-Warnings"])


def test_preview_skips_bypassed_stages_and_rejects_empty_chain():
    r = client.post("/api/preview", json={
        "di": "riff.wav", "stages": [{"type": "nam", "file": "drive.nam", "enabled": False}],
    })
    assert r.status_code == 400 and "vazia" in r.json()["detail"]


def test_preview_without_di_needs_training_signal():
    if server.V3.exists():
        pytest.skip("workspace já tem v3")
    r = client.post("/api/preview", json={"stages": [{"type": "nam", "file": "drive.nam"}]})
    assert r.status_code == 409


def test_job_requires_training_signal():
    r = client.post("/api/jobs", json={"stages": [{"type": "nam", "file": "drive.nam"}]})
    assert r.status_code == 409


def test_create_job_snapshots_inputs(monkeypatch):
    monkeypatch.setattr(server, "_v3_status", lambda: {"valid": True})
    # Não deixa o worker de verdade pegar o job.
    monkeypatch.setattr(server, "_wake", threading.Event())
    r = client.post("/api/jobs", json={
        "name": "Teste", "arch": "nano", "epochs": 3,
        "stages": [
            {"type": "nam", "file": "drive.nam", "output_gain_db": -2},
            {"type": "ir", "file": "cab.wav", "enabled": False},
        ],
    })
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "queued" and len(job["stages"]) == 1
    d = server.JOBS / job["id"]
    chain = json.loads((d / "chain.json").read_text())
    assert chain["arch"] == "nano"
    assert [s["file"] for s in chain["stages"]] == ["inputs/drive.nam"]
    assert (d / "inputs" / "drive.nam").is_file()

    assert client.get(f"/api/jobs/{job['id']}").json()["name"] == "Teste"
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "canceled"
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_job_ids_are_validated():
    assert client.get("/api/jobs/../../etc").status_code == 404
    assert client.get("/api/jobs/zzzzzzzzzzzz").status_code == 404


def test_static_ui_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "Chain Capture" in r.text
