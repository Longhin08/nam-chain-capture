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

client = TestClient(server.app, base_url="http://127.0.0.1")


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
    ir = {f["name"]: f for f in d["ir"]}
    assert ir["cab.wav"]["seconds"] == pytest.approx(64 / 48000, abs=1e-3)
    di = {f["name"]: f for f in d["di"]}
    assert di["riff.wav"]["seconds"] == pytest.approx(3.0)


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


def test_rejects_foreign_host_header():
    other = TestClient(server.app, base_url="http://evil.example")
    assert other.get("/api/status").status_code == 403


def test_token_required_when_configured(monkeypatch):
    monkeypatch.setattr(server, "TOKEN", "segredo")
    assert client.get("/api/files").status_code == 401
    assert client.get("/api/files", headers={"X-Chaincap-Token": "errado"}).status_code == 401
    assert client.get("/api/files", headers={"X-Chaincap-Token": "segredo"}).status_code == 200
    assert client.get("/api/files?token=segredo").status_code == 200
    assert client.get("/").status_code == 200  # a página em si não precisa


def test_status_reports_gpu_runtime():
    s = client.get("/api/status").json()
    assert set(s["gpu_runtime"]) >= {"supported", "nvidia_detected", "installed", "installing"}


def test_worker_command_uses_desktop_entrypoint():
    assert server.worker_base()[-1].endswith("desktop_main.py")


def test_builtin_cab_is_seeded_once():
    names = [f["name"] for f in client.get("/api/files").json()["ir"]]
    assert server.BUILTIN_CAB in names
    assert client.delete(f"/api/files/ir/{server.BUILTIN_CAB}").status_code == 200
    server._seed_builtin_cab()  # próxima abertura do app
    names = [f["name"] for f in client.get("/api/files").json()["ir"]]
    assert server.BUILTIN_CAB not in names  # apagou, não volta


def test_status_exposes_cab_hints():
    s = client.get("/api/status").json()
    assert s["builtin_cab"] == server.BUILTIN_CAB and "amp" in s["amp_only_gear"]


def test_builtin_demo_di_is_seeded_and_previewable():
    di = {f["name"]: f for f in client.get("/api/files").json()["di"]}
    assert server.BUILTIN_DI in di
    assert di[server.BUILTIN_DI]["sample_rate"] == 48000
    assert di[server.BUILTIN_DI]["seconds"] > 15
    r = client.post("/api/preview", json={
        "di": server.BUILTIN_DI, "seconds": 3,
        "stages": [{"type": "nam", "file": "drive.nam"}],
    })
    assert r.status_code == 200 and len(r.content) > 3 * 48000 * 2


def test_preview_resamples_44k_di():
    x = (np.sin(2 * np.pi * 220 * np.arange(44100 * 2) / 44100) * 0.3).astype(np.float32)
    assert upload("di", "loop44.wav", wav_bytes(x, sr=44100)).status_code == 200
    r = client.post("/api/preview", json={
        "di": "loop44.wav", "seconds": 1, "stages": [{"type": "ir", "file": "cab.wav"}],
    })
    assert r.status_code == 200, r.text
    sr, y = wavfile.read(io.BytesIO(r.content))
    assert sr == 48000 and len(y) == 48000
    r = client.get("/api/di-clip?di=loop44.wav&seconds=1", headers={})
    assert r.status_code == 200


def test_all_builtin_dis_seeded_and_legacy_riff_removed(tmp_path, monkeypatch):
    di_dir = server.FILES / "di"
    (di_dir / server._LEGACY_DEMO_DI).write_bytes(b"x")  # instalação antiga
    (server.WS / ".builtin-dis.json").unlink(missing_ok=True)
    server._seed_builtin_di()
    names = {f["name"] for f in client.get("/api/files").json()["di"]}
    assert server._LEGACY_DEMO_DI not in names
    expected = {p.name for p in server.DEMO_DI_DIR.glob("*.wav")}
    assert len(expected) == 3 and expected <= names
    for f in client.get("/api/files").json()["di"]:
        if f["name"] in expected:
            assert f["sample_rate"] == 48000 and f["seconds"] > 9
