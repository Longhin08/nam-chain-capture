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

# Índice com todas as variantes do torch (cpu, cuXXX, rocm, xpu...). Não usamos
# lista fixa de versões de CUDA: o PyTorch troca de CUDA a cada versão.
INDEX = "https://download.pytorch.org/whl/torch/"
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


class DriverTooOld(RuntimeError):
    pass


def _nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi") or os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"
    )
    return exe if os.path.exists(exe) else None


def driver_cuda_version() -> Optional[Tuple[int, int]]:
    """Maior versão de CUDA que o driver instalado suporta (cabeçalho do nvidia-smi)."""
    exe = _nvidia_smi()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe], capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def cuda_tag_version(tag: str) -> Tuple[int, int]:
    """'cu132' -> (13, 2); 'cu118' -> (11, 8); 'cu92' -> (9, 2)."""
    d = tag[2:]
    return int(d[:-1]), int(d[-1])


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


def find_wheel(
    torch_version: str, py_tag: str, plat: str, max_cuda: Optional[Tuple[int, int]] = None
) -> Tuple[str, str, str]:
    """(url, sha256, cuda_tag) do wheel CUDA desta versão exata do torch.

    Entre as variantes publicadas, escolhe a de CUDA mais novo que o driver
    suporta (`max_cuda`, lido do nvidia-smi). Sem `max_cuda`, a mais nova.
    """
    with urllib.request.urlopen(INDEX, timeout=60) as r:
        page = r.read().decode("utf-8", "replace")
    p = _Links()
    p.feed(page)
    pat = re.compile(
        rf"torch-{re.escape(torch_version)}\+(cu\d+)-{py_tag}-{py_tag}-{re.escape(plat)}\.whl"
    )
    found = {}
    for href in p.hrefs:
        url, _, frag = href.partition("#")
        m = pat.fullmatch(urllib.parse.unquote(url.rsplit("/", 1)[-1]))
        sha = re.match(r"sha256=([0-9a-f]{64})", frag)
        if m and sha:
            found[m.group(1)] = (urllib.parse.urljoin(INDEX, url), sha.group(1))
    if not found:
        raise LookupError(
            f"Nenhum PyTorch CUDA {torch_version} para {py_tag}/{plat} no índice oficial"
        )
    tags = sorted(found, key=cuda_tag_version)
    ok = [t for t in tags if max_cuda is None or cuda_tag_version(t) <= max_cuda]
    if not ok:
        need = ".".join(map(str, cuda_tag_version(tags[0])))
        have = ".".join(map(str, max_cuda))
        raise DriverTooOld(
            f"O driver da sua NVIDIA suporta até CUDA {have}, e este PyTorch precisa de "
            f"CUDA {need} ou mais novo. Atualize o driver da NVIDIA (site da NVIDIA ou "
            "GeForce Experience / NVIDIA App) e tente de novo."
        )
    tag = ok[-1]
    url, sha = found[tag]
    return url, sha, tag


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


def _log(workspace: Path, msg: str) -> None:
    """Log da instalação em <dados>/gpu-runtime/install.log (para diagnóstico)."""
    try:
        from datetime import datetime

        _root(workspace).mkdir(parents=True, exist_ok=True)
        with open(_root(workspace) / "install.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    except OSError:
        pass


def gpu_check(worker_cmd: List[str], runtime_path: Path, cwd: Path) -> dict:
    env = dict(os.environ, CHAINCAP_GPU_RUNTIME=str(runtime_path))
    out = subprocess.run(
        # 20 min: na 1ª vez o antivírus analisa ~4 GB de DLLs novas.
        worker_cmd + ["--gpu-check"], capture_output=True, text=True, timeout=1200, env=env, cwd=cwd,
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
        url, sha, tag = find_wheel(tv, py, plat, max_cuda=driver_cuda_version())
        _log(workspace, f"torch {tv} -> {tag}: {url}")
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
        import traceback

        _log(workspace, traceback.format_exc())
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
