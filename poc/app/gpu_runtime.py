"""Pacote opcional de GPU NVIDIA (Windows).

O app é distribuído com o PyTorch CPU (instalador leve). Se houver uma placa
NVIDIA, o usuário pode baixar o PyTorch CUDA oficial *da mesma versão* para a
pasta de dados do app. O processo de treino (worker) ativa esse runtime antes
de importar `torch`; a interface continua no PyTorch CPU embutido.

Fluxo:
    status()   -> o que a interface mostra (placa detectada, instalado, progresso)
    install()  -> descobre o wheel no índice oficial, baixa, confere sha256,
                  extrai, roda `--gpu-check` e grava runtime.json
    activate() -> chamado no worker antes de qualquer `import torch`
"""

import hashlib
import html.parser
import importlib.machinery
import importlib.abc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

INDEX = "https://download.pytorch.org/whl/{tag}/torch/"
# Do mais novo para o mais velho; usamos o primeiro que tiver a versão exata.
CUDA_TAGS = ("cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118")
_TOP_LEVEL = ("torch", "torchgen", "functorch")

_lock = threading.Lock()
_state = {"installing": False, "phase": None, "done_bytes": 0, "total_bytes": 0, "error": None}


def _root(workspace: Path) -> Path:
    return Path(workspace) / "gpu-runtime"


def _marker(workspace: Path) -> Path:
    return _root(workspace) / "runtime.json"


def bundled_torch_version() -> str:
    """Versão do torch embutido, sem o sufixo local (+cpu)."""
    from importlib.metadata import version

    return version("torch").split("+")[0]


def _platform_tag() -> Optional[str]:
    if sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64"):
        return "win_amd64"
    return None  # Linux precisa dos pacotes nvidia-* separados; macOS usa MPS.


def nvidia_gpu_detected() -> bool:
    if sys.platform != "win32":
        return False
    exe = shutil.which("nvidia-smi") or os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"
    )
    if not os.path.exists(exe):
        return False
    try:
        out = subprocess.run(
            [exe, "-L"], capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


def read_marker(workspace: Path) -> Optional[dict]:
    try:
        m = json.loads(_marker(workspace).read_text())
    except (OSError, ValueError):
        return None
    if m.get("torch_version") != bundled_torch_version():
        return None  # app atualizado com outro torch: runtime antigo não serve
    if not Path(m.get("path", "")).is_dir():
        return None
    return m


def status(workspace: Path) -> dict:
    m = read_marker(workspace)
    return {
        "supported": _platform_tag() is not None,
        "nvidia_detected": nvidia_gpu_detected() if _platform_tag() else False,
        "installed": m is not None,
        "device": m.get("device") if m else None,
        "cuda_tag": m.get("cuda_tag") if m else None,
        **_state,
    }


# --------------------------------------------------------------- descoberta


class _Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def find_wheel(torch_version: str, py_tag: str, plat: str) -> Tuple[str, str, str]:
    """Retorna (url, sha256, cuda_tag) do wheel CUDA para esta versão exata."""
    for tag in CUDA_TAGS:
        index = INDEX.format(tag=tag)
        try:
            with urllib.request.urlopen(index, timeout=30) as r:
                page = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        p = _Links()
        p.feed(page)
        want = f"torch-{torch_version}+{tag}-{py_tag}-{py_tag}-{plat}.whl"
        for href in p.hrefs:
            url, _, frag = href.partition("#")
            if urllib.parse.unquote(url.rsplit("/", 1)[-1]) == want:
                m = re.match(r"sha256=([0-9a-f]{64})", frag)
                if not m:
                    continue
                return urllib.parse.urljoin(index, url), m.group(1), tag
    raise LookupError(
        f"Nenhum PyTorch CUDA {torch_version} para {py_tag}/{plat} no índice oficial"
    )


# ------------------------------------------------------------- instalação


def _download(url: str, sha256: str, dest: Path) -> None:
    h = hashlib.sha256()
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        _state["total_bytes"] = int(r.headers.get("Content-Length") or 0)
        _state["done_bytes"] = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            _state["done_bytes"] += len(chunk)
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError("O arquivo baixado não confere com o sha256 do índice oficial")
    tmp.replace(dest)


def _extract(wheel: Path, target: Path) -> None:
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            top = info.filename.split("/", 1)[0]
            if top in _TOP_LEVEL or top.endswith(".dist-info"):
                out = (target / info.filename).resolve()
                if not str(out).startswith(str(target.resolve())):
                    raise ValueError(f"Caminho inválido no wheel: {info.filename}")
                z.extract(info, target)


def gpu_check(worker_cmd: List[str], runtime_path: Path, cwd: Path) -> dict:
    env = dict(os.environ, CHAINCAP_GPU_RUNTIME=str(runtime_path))
    out = subprocess.run(
        worker_cmd + ["--gpu-check"], capture_output=True, text=True, timeout=300, env=env, cwd=cwd,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"Teste da GPU falhou: {(out.stderr or out.stdout)[-400:]}")
    return json.loads(lines[-1])


def _install(workspace: Path, worker_cmd: List[str]) -> None:
    root = _root(workspace)
    try:
        plat = _platform_tag()
        if plat is None:
            raise RuntimeError("Pacote de GPU só existe para Windows 64 bits")
        tv = bundled_torch_version()
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        _state.update(phase="procurando", error=None)
        url, sha, tag = find_wheel(tv, py, plat)
        root.mkdir(parents=True, exist_ok=True)
        wheel = root / urllib.parse.unquote(url.rsplit("/", 1)[-1])
        _state["phase"] = "baixando"
        _download(url, sha, wheel)
        _state["phase"] = "instalando"
        target = root / f"torch-{tv}+{tag}"
        if target.exists():
            shutil.rmtree(target)
        _extract(wheel, target)
        wheel.unlink(missing_ok=True)
        _state["phase"] = "testando"
        check = gpu_check(worker_cmd, target, workspace)
        if not check.get("cuda"):
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(
                "O PyTorch CUDA foi instalado mas não encontrou a placa. "
                "Atualize o driver da NVIDIA e tente de novo."
            )
        _marker(workspace).write_text(json.dumps({
            "torch_version": tv, "cuda_tag": tag, "path": str(target), "device": check.get("name"),
        }))
        for old in root.iterdir():  # limpa versões antigas
            if old.is_dir() and old != target:
                shutil.rmtree(old, ignore_errors=True)
    except Exception as e:
        _state["error"] = str(e)
    finally:
        _state.update(installing=False, phase=None)


def install(workspace: Path, worker_cmd: List[str]) -> dict:
    with _lock:
        if not _state["installing"]:
            _state.update(installing=True, error=None, done_bytes=0, total_bytes=0)
            threading.Thread(target=_install, args=(Path(workspace), worker_cmd), daemon=True).start()
    return status(workspace)


def remove(workspace: Path) -> dict:
    if _state["installing"]:
        raise RuntimeError("Aguarde a instalação terminar")
    shutil.rmtree(_root(workspace), ignore_errors=True)
    return status(workspace)


# --------------------------------------------------------------- ativação


class _RuntimeFinder(importlib.abc.MetaPathFinder):
    """Faz `import torch` (e submódulos) vir do runtime baixado, antes do embutido."""

    def __init__(self, path: Path):
        self.path = str(path)

    def find_spec(self, name, path=None, target=None):
        top = name.split(".", 1)[0]
        if top not in _TOP_LEVEL:
            return None
        search = [self.path] if "." not in name else path
        return importlib.machinery.PathFinder.find_spec(name, search)


def activate(workspace: Optional[Path] = None) -> Optional[str]:
    """Ativa o runtime de GPU no processo atual. Chamar antes de `import torch`."""
    if "torch" in sys.modules:
        raise RuntimeError("activate() precisa rodar antes de importar torch")
    path = os.environ.get("CHAINCAP_GPU_RUNTIME")
    if not path and workspace is not None:
        m = read_marker(workspace)
        path = m["path"] if m else None
    if not path or not Path(path, "torch").is_dir():
        return None
    sys.meta_path.insert(0, _RuntimeFinder(Path(path)))
    return path
