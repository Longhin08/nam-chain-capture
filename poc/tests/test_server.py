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


def test_list_files_describes_each_kind():
    d = client.get("/api/files").json()
    assert [f["name"] for f in d["nam"]] == ["drive.nam"]
    assert d["nam"][0]["architecture"] == "WaveNet"
    assert d["nam"][0]["input_level_dbu"] == 12.0
    ir = {f["name"]: f for f in d["ir"]}
    assert ir["cab.wav"]["seconds"] == pytest.approx(64 / 48000, abs=1e-3)
    assert {f["name"] for f in d["di"] if f.get("builtin")} == {p.name for p in server.DEMO_DI_DIR.glob("*.wav")}


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
        "seconds": 2,
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
        "stages": [{"type": "nam", "file": "drive.nam", "enabled": False}],
    })
    assert r.status_code == 400 and "vazia" in r.json()["detail"]


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



def test_preview_uses_only_builtin_test_audio():
    names = [f["name"] for f in client.get("/api/files").json()["di"] if f.get("builtin")]
    assert server.BUILTIN_DI in names and len(names) == 3
    for f in client.get("/api/files").json()["di"]:
        if not f.get("builtin"):
            continue
        assert f["sample_rate"] == 48000 and f["seconds"] > 9
    # sem escolher: usa o padrão
    r = client.post("/api/preview", json={"seconds": 2, "stages": [{"type": "nam", "file": "drive.nam"}]})
    assert r.status_code == 200 and len(r.content) > 2 * 48000 * 2
    # cada um dos embutidos funciona
    for n in names:
        assert client.get(f"/api/di-clip?di={n}&seconds=1").status_code == 200
    # nada fora da lista
    assert client.get("/api/di-clip?di=../../v3_0_0.wav&seconds=1").status_code == 404
    assert client.get("/api/di-clip?di=outro.wav").status_code == 404  # não existe


SPLIT = {"type": "split", "branches": [
    {"stages": [{"type": "nam", "file": "drive.nam"}], "level_db": -3},
    {"stages": [{"type": "ir", "file": "cab.wav"}], "invert": True},
]}


def test_preview_with_split():
    r = client.post("/api/preview", json={"seconds": 1, "stages": [SPLIT]})
    assert r.status_code == 200, r.text


def test_api_rejects_two_splits_and_empty_split():
    r = client.post("/api/preview", json={"seconds": 1, "stages": [SPLIT, SPLIT]})
    assert r.status_code == 400 and "Só uma divisão" in r.json()["detail"]
    empty = {"type": "split", "branches": [{"stages": []}, {"stages": []}]}
    r = client.post("/api/preview", json={"seconds": 1, "stages": [empty]})
    assert r.status_code == 400 and "vazia" in r.json()["detail"]
    nested = {"type": "split", "branches": [{"stages": [SPLIT]}, {"stages": []}]}
    assert client.post("/api/preview", json={"seconds": 1, "stages": [nested]}).status_code == 422


def test_job_with_split_snapshots_branch_files(monkeypatch):
    monkeypatch.setattr(server, "_v3_status", lambda: {"valid": True})
    monkeypatch.setattr(server, "_wake", threading.Event())
    r = client.post("/api/jobs", json={"name": "Dual", "arch": "nano", "epochs": 1,
                                       "stages": [{"type": "nam", "file": "drive.nam"}, SPLIT]})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    chain = json.loads((server.JOBS / jid / "chain.json").read_text())
    split = chain["stages"][1]
    assert split["type"] == "split" and len(split["branches"]) == 2
    files = [chain["stages"][0]["file"]] + [st["file"] for b in split["branches"] for st in b["stages"]]
    for f in files:
        assert (server.JOBS / jid / f).is_file()
    assert split["branches"][1]["invert"] is True
    assert r.json()["has_split"] is True
    client.post(f"/api/jobs/{jid}/cancel")
    client.delete(f"/api/jobs/{jid}")


def test_user_di_upload_is_back_and_listed_after_builtins():
    x = (np.sin(2 * np.pi * 220 * np.arange(44100) / 44100) * 0.3).astype(np.float32)
    builtin = server.BUILTIN_DI
    r = upload("di", builtin, wav_bytes(x, sr=44100))  # mesmo nome de um embutido
    assert r.status_code == 200 and r.json()["name"] != builtin
    mine = r.json()["name"]
    di = client.get("/api/files").json()["di"]
    assert [f.get("builtin", False) for f in di][:3] == [True, True, True]
    assert di[-1]["name"] == mine
    r = client.post("/api/preview", json={"di": mine, "seconds": 0.5, "stages": [{"type": "ir", "file": "cab.wav"}]})
    assert r.status_code == 200  # reamostrado de 44,1 para 48 kHz
    assert client.delete(f"/api/files/di/{builtin}").status_code == 404  # embutido não apaga
    assert client.delete(f"/api/files/di/{mine}").status_code == 200


def test_ping_and_autostop_rules():
    import desktop_main as D

    assert client.post("/api/ping").status_code == 200
    assert server.LAST_PING["t"] is not None
    # sem ping ainda: espera o período de carência
    assert not D.should_autostop(None, 0, D.GRACE_S - 1, False)
    assert D.should_autostop(None, 0, D.GRACE_S + 1, False)
    # aba fechada: para depois de AUTOSTOP_S sem ping
    assert not D.should_autostop(100, 0, 100 + D.AUTOSTOP_S - 1, False)
    assert D.should_autostop(100, 0, 100 + D.AUTOSTOP_S + 1, False)
    # com treino rodando, nunca para sozinho
    assert not D.should_autostop(100, 0, 10_000, True)


def test_blocked_worker_marks_job_failed_and_queue_survives(monkeypatch):
    """Se o Windows barrar o processo de treino, o job falha com mensagem clara
    (em vez de ficar 'Treinando' para sempre) e a fila continua andando."""
    monkeypatch.setattr(server, "_v3_status", lambda: {"valid": True})
    monkeypatch.setattr(server, "_wake", threading.Event())  # sem worker real

    def blocked(*a, **k):
        e = OSError("An Application Control policy has blocked this file")
        e.winerror = 4551
        raise e

    monkeypatch.setattr(server.subprocess, "Popen", blocked)
    r = client.post("/api/jobs", json={"name": "Bloq", "arch": "nano", "epochs": 1,
                                       "stages": [{"type": "nam", "file": "drive.nam"}]})
    jid = r.json()["id"]
    with server._jobs_lock:
        server._queue.remove(jid)
    server._run_job(jid)
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "failed"
    assert "bloqueou" in job["error"] and "portátil" in job["error"]
    assert server._running["id"] is None
    # erro genérico ao iniciar também vira falha
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("sem permissão")))
    r = client.post("/api/jobs", json={"name": "X", "arch": "nano", "epochs": 1,
                                       "stages": [{"type": "nam", "file": "drive.nam"}]})
    jid2 = r.json()["id"]
    with server._jobs_lock:
        server._queue.remove(jid2)
    server._run_job(jid2)
    assert "sem permissão" in client.get(f"/api/jobs/{jid2}").json()["error"]
    for j in (jid, jid2):
        client.delete(f"/api/jobs/{j}")
