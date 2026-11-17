"""Interface local para montar a cadeia de capturas e treinar.

    cd poc && ../.venv/bin/python -m uvicorn app.server:app --port 8765

Tudo fica em `poc/workspace/` (ou `$CHAINCAP_WORKSPACE`):
    files/nam/*.nam   capturas
    files/ir/*.wav    IRs
    files/di/*.wav    DIs para preview
    v3_0_0.wav        sinal de treino oficial
    jobs/<id>/        chain.json, log.txt, job.json, out/
"""

import chaincap  # noqa: F401  (shim do tkinter antes de qualquer nam)

import hmac
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.io import wavfile

from chaincap.archs import ARCHITECTURES
from chaincap.chain import (
    SAMPLE_RATE,
    Branch,
    Split,
    Stage,
    load_ir,
    read_nam_file,
    read_wav_mono,
    render_chain,
)
from chaincap.train import V3_MD5, md5

from . import gpu_runtime

FROZEN = getattr(sys, "frozen", False)
POC_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Token por sessão (o app desktop gera um a cada abertura). Sem ele, a API recusa.
TOKEN = os.environ.get("CHAINCAP_TOKEN") or None
WS = Path(os.environ.get("CHAINCAP_WORKSPACE", POC_DIR / "workspace")).resolve()
FILES = WS / "files"
JOBS = WS / "jobs"
V3 = WS / "v3_0_0.wav"
V3_DOWNLOAD = (
    "https://drive.usercontent.google.com/download?"
    "id=1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9&export=download&confirm=t"
)
KINDS = ("nam", "ir", "di")
MAX_UPLOAD = 64 * 1024 * 1024
PREVIEW_MAX_S = 30.0
# IR de caixa gerado pelo app (ver chaincap/cab_ir.py), livre para distribuir.
BUILTIN_CAB = "Caixa 4x12 genérica.wav"
# Áudios de teste do preview: gravações CC0 do Freesound embutidas no app (ver
# packaging/make_demo_dis.py), mais os DIs que o usuário enviar (só locais).
DEMO_DI_DIR = Path(__file__).resolve().parent / "assets" / "di"
BUILTIN_DI = "Strat limpa - loop (mil_dew).wav"  # padrão do preview
# Tipos de captura sem caixa: precisam de IR depois para não soarem ásperos.
AMP_ONLY_GEAR = ("amp", "pedal_amp", "preamp")

for k in KINDS:
    (FILES / k).mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)


def _seed_builtin_cab() -> None:
    """Coloca a caixa genérica na biblioteca uma vez (se o usuário apagar, não volta)."""
    marker = WS / ".builtin-cab-seeded"
    if marker.exists():
        return
    from chaincap.cab_ir import write_cab_ir

    dest = FILES / "ir" / BUILTIN_CAB
    if not dest.exists():
        write_cab_ir(dest)
    marker.write_text("1")


_seed_builtin_cab()

app = FastAPI(title="NAM Chain Capture")


def worker_base() -> List[str]:
    """Comando que roda o worker (treino / teste de GPU) neste ambiente."""
    if FROZEN:
        return [sys.executable]
    return [sys.executable, str(POC_DIR / "desktop_main.py")]


_LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


@app.middleware("http")
async def _guard(request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host not in _LOCAL_HOSTS:
        # Bloqueia DNS rebinding: só aceitamos o próprio endereço local.
        return JSONResponse({"detail": "Host não permitido"}, status_code=403)
    if TOKEN and request.url.path.startswith("/api/"):
        got = request.headers.get("x-chaincap-token") or request.query_params.get("token") or ""
        if not hmac.compare_digest(got, TOKEN):
            return JSONResponse({"detail": "Sessão inválida; reabra o app"}, status_code=401)
    return await call_next(request)


# ------------------------------------------------------------------ arquivos


def _safe_name(name: str) -> str:
    name = Path(name).name
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^\w\-. ()]+", "_", stem).strip(" .") or "arquivo"
    return f"{stem[:80]}{ext.lower()}"


def _unique(dirpath: Path, name: str) -> Path:
    p = dirpath / name
    stem, ext = os.path.splitext(name)
    i = 2
    while p.exists():
        p = dirpath / f"{stem} ({i}){ext}"
        i += 1
    return p


def _file_path(kind: str, name: str) -> Path:
    if kind not in KINDS:
        raise HTTPException(400, f"Tipo inválido: {kind}")
    p = (FILES / kind / Path(name).name).resolve()
    if p.parent != (FILES / kind).resolve() or not p.is_file():
        raise HTTPException(404, f"Arquivo não encontrado: {name}")
    return p


def _describe(kind: str, p: Path) -> dict:
    info = {"kind": kind, "name": p.name, "bytes": p.stat().st_size}
    try:
        if kind == "nam":
            d = read_nam_file(p)
            md = d.get("metadata") or {}
            info.update(
                architecture=d.get("architecture"),
                version=d.get("version"),
                sample_rate=d.get("sample_rate"),
                title=md.get("name"),
                modeled_by=md.get("modeled_by"),
                gear_type=md.get("gear_type"),
                gear_make=md.get("gear_make"),
                gear_model=md.get("gear_model"),
                tone_type=md.get("tone_type"),
                input_level_dbu=md.get("input_level_dbu"),
                weights=len(d.get("weights") or []),
            )
        else:
            x, sr = read_wav_mono(p)
            info.update(sample_rate=sr, seconds=round(len(x) / sr, 3))
    except Exception as e:  # arquivo corrompido: mostra mas marca erro
        info["error"] = str(e)
    return info


def _demo_dis() -> List[Path]:
    return sorted(DEMO_DI_DIR.glob("*.wav")) if DEMO_DI_DIR.is_dir() else []


@app.get("/api/files")
def list_files():
    out = {
        k: [_describe(k, p) for p in sorted((FILES / k).iterdir()) if p.is_file()]
        for k in KINDS
    }
    builtin = [{**_describe("di", p), "builtin": True} for p in _demo_dis()]
    out["di"] = builtin + out["di"]
    return out


@app.post("/api/files")
async def upload(kind: Literal["nam", "ir", "di"] = Form(...), file: UploadFile = File(...)):
    name = _safe_name(file.filename or "arquivo")
    ext = os.path.splitext(name)[1]
    if kind == "nam" and ext != ".nam":
        raise HTTPException(400, "Capturas precisam ser arquivos .nam")
    if kind in ("ir", "di") and ext != ".wav":
        raise HTTPException(400, "IRs e DIs precisam ser arquivos .wav")
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Arquivo maior que 64 MB")
    dest = _unique(FILES / kind, name)
    if kind == "di":  # não pode ter o mesmo nome de um áudio embutido
        builtin = {p.name for p in _demo_dis()}
        stem_, ext_ = os.path.splitext(dest.name)
        i = 2
        while dest.name in builtin or dest.exists():
            dest = FILES / "di" / f"{stem_} ({i}){ext_}"
            i += 1
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    try:
        if kind == "nam":
            read_nam_file(tmp)
        elif kind == "ir":
            load_ir(tmp)
        else:
            read_wav_mono(tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"{name}: {e}")
    tmp.rename(dest)
    return _describe(kind, dest)


@app.delete("/api/files/{kind}/{name}")
def delete_file(kind: str, name: str):
    _file_path(kind, name).unlink()
    return {"ok": True}


# ------------------------------------------------------------------- cadeia


class StageIn(BaseModel):
    type: Literal["nam", "ir"]
    file: Optional[str] = None  # nome em files/<type>/
    job: Optional[str] = None  # alternativa: modelo treinado de um job
    input_gain_db: float = Field(0.0, ge=-48, le=48)
    output_gain_db: float = Field(0.0, ge=-48, le=48)
    ir_max_ms: Optional[float] = Field(None, gt=0, le=2000)
    ir_normalize: bool = True
    enabled: bool = True


class BranchIn(BaseModel):
    stages: List[StageIn] = []
    level_db: float = Field(0.0, ge=-48, le=12)
    invert: bool = False
    enabled: bool = True


class SplitIn(BaseModel):
    type: Literal["split"]
    branches: List[BranchIn] = Field(..., min_length=2, max_length=4)
    enabled: bool = True  # (não usado: uma divisão não tem bypass próprio)


NodeIn = Annotated[Union[StageIn, SplitIn], Field(discriminator="type")]


def _to_stage(s: StageIn) -> Stage:
    if s.job:
        if s.type != "nam":
            raise HTTPException(400, "Só estágios NAM podem vir de um job")
        p = JOBS / _job_id(s.job) / "out" / "chain.nam"
        if not p.is_file():
            raise HTTPException(404, f"Job {s.job} ainda não tem modelo")
    elif s.file:
        p = _file_path(s.type, s.file)
    else:
        raise HTTPException(400, "Estágio sem arquivo")
    return Stage(
        type=s.type,
        file=p,
        input_gain_db=s.input_gain_db,
        output_gain_db=s.output_gain_db,
        ir_max_ms=s.ir_max_ms,
        ir_normalize=s.ir_normalize,
    )


def _active(nodes: List[NodeIn]):
    """Converte o que veio da interface em nós do motor, pulando o que está em bypass."""
    out, n_stages = [], 0
    if sum(isinstance(n, SplitIn) for n in nodes) > 1:
        raise HTTPException(400, "Só uma divisão por cadeia")
    for n in nodes:
        if isinstance(n, SplitIn):
            branches = []
            for b in n.branches:
                st = [_to_stage(s) for s in b.stages if s.enabled]
                if b.enabled:
                    n_stages += len(st)
                branches.append(Branch(stages=st, level_db=b.level_db, invert=b.invert, enabled=b.enabled))
            if not any(b.enabled for b in branches):
                raise HTTPException(400, "Todas as cadeias da divisão estão no mudo")
            out.append(Split(branches=branches))
        elif n.enabled:
            out.append(_to_stage(n))
            n_stages += 1
    if not n_stages:
        raise HTTPException(400, "A cadeia está vazia (ou todos os estágios em bypass)")
    return out


def _wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    wavfile.write(buf, SAMPLE_RATE, pcm)
    return buf.getvalue()


def _di_clip(di: Optional[str], seconds: float) -> np.ndarray:
    n = int(min(seconds, PREVIEW_MAX_S) * SAMPLE_RATE)
    name = Path(di or BUILTIN_DI).name
    builtin = DEMO_DI_DIR / name
    path = builtin if builtin in _demo_dis() else _file_path("di", name)
    x, sr = read_wav_mono(path)
    if sr != SAMPLE_RATE:
        # DIs baixados costumam vir em 44,1 kHz: reamostra para 48 kHz.
        from scipy.signal import resample_poly

        g = int(np.gcd(sr, SAMPLE_RATE))
        x = resample_poly(x[: int(seconds * sr) + sr], SAMPLE_RATE // g, sr // g).astype(np.float32)
    return x[:n]


class PreviewIn(BaseModel):
    stages: List[NodeIn]
    di: Optional[str] = None
    seconds: float = Field(15.0, gt=0, le=PREVIEW_MAX_S)


@app.post("/api/preview")
def preview(req: PreviewIn):
    x = _di_clip(req.di, req.seconds)
    try:
        res = render_chain(x, _active(req.stages))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
    peak = float(np.max(np.abs(res.y))) / res.clip_gain
    # Normaliza para -1 dBFS só para ouvir; o nível real vai no header.
    y = res.y / max(float(np.max(np.abs(res.y))), 1e-9) * 0.89
    headers = {
        "X-Chain-Peak-Dbfs": f"{20 * np.log10(max(peak, 1e-9)):.1f}",
        "X-Chain-Warnings": json.dumps(res.warnings, ensure_ascii=True),
    }
    return Response(_wav_bytes(y), media_type="audio/wav", headers=headers)


@app.get("/api/di-clip")
def di_clip(di: Optional[str] = None, seconds: float = 15.0):
    x = _di_clip(di, seconds)
    peak = max(float(np.max(np.abs(x))), 1e-9)
    return Response(_wav_bytes(x / peak * 0.89), media_type="audio/wav")


# --------------------------------------------------------------- sinal v3

_v3_lock = threading.Lock()
_v3_state = {"downloading": False, "error": None}


_v3_md5_cache: dict = {}


def _v3_status() -> dict:
    ok = False
    if V3.is_file():
        key = (V3.stat().st_mtime_ns, V3.stat().st_size)
        if key not in _v3_md5_cache:
            _v3_md5_cache.clear()
            _v3_md5_cache[key] = md5(V3) == V3_MD5
        ok = _v3_md5_cache[key]
    return {"present": V3.is_file(), "valid": ok, **_v3_state}


def _download_v3():
    try:
        tmp = V3.with_suffix(".part")
        with urllib.request.urlopen(V3_DOWNLOAD, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if md5(tmp) != V3_MD5:
            tmp.unlink(missing_ok=True)
            raise ValueError("arquivo baixado não confere com o md5 oficial")
        tmp.replace(V3)
        _v3_state["error"] = None
    except Exception as e:
        _v3_state["error"] = f"Falha ao baixar: {e}"
    finally:
        _v3_state["downloading"] = False


@app.post("/api/v3/download")
def download_v3():
    with _v3_lock:
        if not _v3_state["downloading"] and not V3.is_file():
            _v3_state.update(downloading=True, error=None)
            threading.Thread(target=_download_v3, daemon=True).start()
    return _v3_status()


def _gpu_name() -> Optional[str]:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "Apple (Metal)"
    m = gpu_runtime.read_marker(WS)
    return m.get("device") if m else None


# Modo navegador: a página avisa que está aberta; sem avisos, o app se encerra.
LAST_PING = {"t": None}


@app.post("/api/ping")
def ping():
    LAST_PING["t"] = time.time()
    return {"ok": True}


def jobs_busy() -> bool:
    with _jobs_lock:
        return _running["id"] is not None or bool(_queue)


@app.get("/api/status")
def status():
    return {
        "v3": _v3_status(),
        "gpu": _gpu_name(),
        "gpu_runtime": gpu_runtime.status(WS),
        "archs": list(ARCHITECTURES),
        "builtin_cab": BUILTIN_CAB,
        "builtin_di": BUILTIN_DI,
        "amp_only_gear": list(AMP_ONLY_GEAR),
        "workspace": str(WS),
        "desktop": FROZEN or bool(os.environ.get("CHAINCAP_DESKTOP")),
    }


@app.post("/api/gpu/install")
def gpu_install():
    if not gpu_runtime.status(WS)["supported"]:
        raise HTTPException(400, "Pacote de GPU só existe para Windows 64 bits")
    return gpu_runtime.install(WS, worker_base())


@app.post("/api/gpu/remove")
def gpu_remove():
    try:
        return gpu_runtime.remove(WS)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


# --------------------------------------------------------------------- jobs

_jobs_lock = threading.Lock()
_queue: List[str] = []
_running: dict = {"id": None, "proc": None}
_wake = threading.Event()


def _job_id(jid: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{12}", jid):
        raise HTTPException(404, "Job inválido")
    if not (JOBS / jid).is_dir():
        raise HTTPException(404, "Job não encontrado")
    return jid


def _read_job(jid: str) -> dict:
    d = JOBS / jid
    job = json.loads((d / "job.json").read_text())
    for name in ("progress.json", "report.json"):
        p = d / "out" / name
        if p.is_file():
            try:
                job[name.split(".")[0]] = json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
    outs = d / "out"
    job["files"] = sorted(
        f.name for f in outs.glob("*") if f.suffix in (".nam", ".json", ".wav") and f.name != "progress.json"
    ) if outs.is_dir() else []
    return job


def _write_job(jid: str, **upd) -> dict:
    p = JOBS / jid / "job.json"
    job = json.loads(p.read_text())
    job.update(upd)
    p.write_text(json.dumps(job, indent=2, ensure_ascii=False))
    return job


# Códigos do Windows quando uma política (Controle Inteligente de Aplicativos,
# WDAC, AppLocker) impede um programa de rodar.
_WIN_POLICY_BLOCKED = {4551, 1260}


def _start_failure_message(e: OSError) -> str:
    if getattr(e, "winerror", None) in _WIN_POLICY_BLOCKED:
        return (
            "O Windows bloqueou o processo de treino (Controle Inteligente de "
            "Aplicativos ou política de segurança). Use a versão portátil do app, "
            "que treina com o Python oficial assinado."
        )
    return f"Não foi possível iniciar o processo de treino: {e}"


def _run_job(jid: str) -> None:
    d = JOBS / jid
    job = json.loads((d / "job.json").read_text())
    if job["status"] != "queued":
        return
    cmd = worker_base() + [
        "--worker", "run", str(d / "chain.json"),
        "--input", str(V3), "--out", str(d / "out"),
        "--epochs", str(job["epochs"]), "--arch", job["arch"],
    ]
    if job.get("gpu_usd_per_hour"):
        cmd += ["--gpu-usd-per-hour", str(job["gpu_usd_per_hour"])]
    _write_job(jid, status="running", started_at=time.time())
    with open(d / "log.txt", "w") as log:
        try:
            proc = subprocess.Popen(
                cmd, cwd=WS, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CHAINCAP_LOG=str(d / "log.txt")),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            msg = _start_failure_message(e)
            log.write(msg + "\n")
            _write_job(jid, status="failed", error=msg, finished_at=time.time())
            return
        with _jobs_lock:
            _running.update(id=jid, proc=proc)
        rc = proc.wait()
    with _jobs_lock:
        canceled = _read_job(jid)["status"] == "canceling"
        _running.update(id=None, proc=None)
    status = "canceled" if canceled else {0: "done", 2: "rejected"}.get(rc, "failed")
    _write_job(jid, status=status, returncode=rc, finished_at=time.time())


def _worker():
    while True:
        _wake.wait()
        with _jobs_lock:
            jid = _queue.pop(0) if _queue else None
            if jid is None:
                _wake.clear()
                continue
        try:
            _run_job(jid)
        except Exception as e:  # nunca deixar a fila morrer por causa de um job
            with _jobs_lock:
                _running.update(id=None, proc=None)
            try:
                _write_job(jid, status="failed", error=f"Erro interno: {e}", finished_at=time.time())
            except Exception:
                pass


def _recover():
    """Ao subir o servidor: jobs que estavam rodando foram interrompidos; fila é retomada."""
    queued = []
    for d in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime):
        f = d / "job.json"
        if not f.is_file():
            continue
        job = json.loads(f.read_text())
        if job["status"] in ("running", "canceling"):
            _write_job(d.name, status="failed", error="Servidor reiniciado durante o treino")
        elif job["status"] == "queued":
            queued.append(d.name)
    _queue.extend(queued)
    if queued:
        _wake.set()


def shutdown_jobs() -> None:
    """Ao fechar o app: interrompe o treino em andamento (vira 'failed' na próxima abertura)."""
    with _jobs_lock:
        proc = _running.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


_recover()
threading.Thread(target=_worker, daemon=True).start()
if os.environ.get("CHAINCAP_AUTO_V3") and not V3.is_file():
    download_v3()


class JobIn(BaseModel):
    name: str = Field("Minha cadeia", min_length=1, max_length=80)
    arch: Literal[ARCHITECTURES] = "a2"  # type: ignore[valid-type]
    epochs: int = Field(100, ge=1, le=1000)
    gpu_usd_per_hour: Optional[float] = Field(None, ge=0)
    stages: List[NodeIn]


@app.post("/api/jobs")
def create_job(req: JobIn):
    if not _v3_status()["valid"]:
        raise HTTPException(409, "Baixe o sinal de treino (v3_0_0.wav) antes de treinar")
    stages = _active(req.stages)
    jid = uuid.uuid4().hex[:12]
    d = JOBS / jid
    (d / "inputs").mkdir(parents=True)
    # Copia os arquivos: o job continua reprodutível mesmo se a biblioteca mudar.
    def snap(st: Stage) -> dict:
        dest = _unique(d / "inputs", Path(st.file).name)
        shutil.copy2(st.file, dest)
        return {
            "type": st.type,
            "file": f"inputs/{dest.name}",
            "input_gain_db": st.input_gain_db,
            "output_gain_db": st.output_gain_db,
            "ir_max_ms": st.ir_max_ms,
            "ir_normalize": st.ir_normalize,
        }

    chain_stages = []
    for node in stages:
        if isinstance(node, Split):
            chain_stages.append({"type": "split", "branches": [
                {"level_db": b.level_db, "invert": b.invert, "enabled": b.enabled,
                 "stages": [snap(st) for st in b.stages]}
                for b in node.branches
            ]})
        else:
            chain_stages.append(snap(node))
    (d / "chain.json").write_text(json.dumps(
        {"name": req.name, "arch": req.arch, "stages": chain_stages}, indent=2, ensure_ascii=False
    ))
    job = {
        "id": jid,
        "name": req.name,
        "arch": req.arch,
        "epochs": req.epochs,
        "gpu_usd_per_hour": req.gpu_usd_per_hour,
        "stages": [s.model_dump() for s in req.stages if s.enabled],
        "has_split": any(isinstance(s, SplitIn) for s in req.stages),
        "status": "queued",
        "created_at": time.time(),
    }
    (d / "job.json").write_text(json.dumps(job, indent=2, ensure_ascii=False))
    with _jobs_lock:
        _queue.append(jid)
        _wake.set()
    return _read_job(jid)


@app.get("/api/jobs")
def list_jobs():
    jobs = []
    for d in JOBS.iterdir():
        if (d / "job.json").is_file():
            jobs.append(_read_job(d.name))
    return sorted(jobs, key=lambda j: j["created_at"], reverse=True)


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _read_job(_job_id(jid))


@app.get("/api/jobs/{jid}/log")
def job_log(jid: str, tail: int = 80):
    p = JOBS / _job_id(jid) / "log.txt"
    if not p.is_file():
        return {"lines": []}
    lines = p.read_text(errors="replace").replace("\r", "\n").splitlines()
    lines = [l for l in lines if l.strip() and "Warning" not in l and "warn(" not in l]
    return {"lines": lines[-tail:]}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    jid = _job_id(jid)
    with _jobs_lock:
        job = _read_job(jid)
        if job["status"] == "queued":
            if jid in _queue:
                _queue.remove(jid)
            return _write_job(jid, status="canceled")
        if job["status"] == "running" and _running["id"] == jid:
            _write_job(jid, status="canceling")
            _running["proc"].terminate()
            return _read_job(jid)
    raise HTTPException(409, f"Job está {job['status']}")


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    jid = _job_id(jid)
    if _read_job(jid)["status"] in ("running", "canceling", "queued"):
        raise HTTPException(409, "Cancele o job antes de apagar")
    shutil.rmtree(JOBS / jid)
    return {"ok": True}


@app.get("/api/jobs/{jid}/files/{name}")
def job_file(jid: str, name: str):
    p = (JOBS / _job_id(jid) / "out" / Path(name).name).resolve()
    if not p.is_file() or p.parent != (JOBS / jid / "out").resolve():
        raise HTTPException(404, "Arquivo não encontrado")
    return FileResponse(p, filename=p.name)


@app.exception_handler(ValueError)
def _value_error(_, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        # App local: sempre revalidar, para não rodar JS/CSS velho depois de atualizar.
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
