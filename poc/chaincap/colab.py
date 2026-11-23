"""Treino no Google Colab a partir de um pacote `.chaincap` gerado pelo app.

O notebook (`chain_capture_colab.ipynb`) só chama as funções daqui, para a
pessoa não ver código: receber o pacote, baixar o sinal de treino, treinar com
barra de progresso e entregar o `.nam`. Tudo que não depende do Colab
(`google.colab`, GPU) roda e é testado localmente.

Pacote `.chaincap` (zip):
    chain.json      cadeia com caminhos relativos a `files/`
    settings.json   {"format": 1, "name", "arch", "epochs", "warm_start",
                     "auto_stop", "app_version"}
    files/...       capturas .nam e IRs .wav usados na cadeia
"""

import html
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PACKAGE_FORMAT = 1
PACKAGE_SUFFIX = ".chaincap"
V3_MD5 = "36cd1af62985c2fac3e654333e36431e"
V3_URL = ("https://drive.usercontent.google.com/download?"
          "id=1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9&export=download&confirm=t")
MAX_PACKAGE_BYTES = 200 * 2**20


class PackageError(ValueError):
    """Pacote inválido: a mensagem é mostrada para a pessoa."""


def safe_filename(name: str, default: str = "captura") -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name or "").strip() or default


# ------------------------------------------------------------------- pacote


def build_package(dest: Path, chain: Dict[str, Any], files: Dict[str, Path],
                  settings: Dict[str, Any]) -> Path:
    """Cria o `.chaincap` (usado pelo app). `files`: nome no pacote -> arquivo local."""
    dest = Path(dest)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("chain.json", json.dumps(chain, indent=2, ensure_ascii=False))
        z.writestr("settings.json", json.dumps({"format": PACKAGE_FORMAT, **settings},
                                               indent=2, ensure_ascii=False))
        for arc, src in files.items():
            z.write(src, f"files/{arc}")
    return dest


def open_package(pkg: Path, workdir: Path) -> Dict[str, Any]:
    """Extrai e confere o pacote. Devolve as configurações + caminhos."""
    pkg, workdir = Path(pkg), Path(workdir)
    if not zipfile.is_zipfile(pkg):
        raise PackageError(
            "Esse arquivo não é um pacote do NAM Chain Capture. No app, clique em "
            "\"Treinar grátis na nuvem\" e envie o arquivo .chaincap que ele salvou."
        )
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    with zipfile.ZipFile(pkg) as z:
        total = 0
        for info in z.infolist():
            total += info.file_size
            target = (workdir / info.filename).resolve()
            if workdir.resolve() not in target.parents and target != workdir.resolve():
                raise PackageError("Pacote com caminhos inválidos.")
        if total > MAX_PACKAGE_BYTES:
            raise PackageError("Pacote grande demais.")
        z.extractall(workdir)
    try:
        settings = json.loads((workdir / "settings.json").read_text(encoding="utf-8"))
        chain = json.loads((workdir / "chain.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PackageError("Pacote incompleto: gere de novo pelo app.")
    if settings.get("format") != PACKAGE_FORMAT:
        raise PackageError("Pacote de outra versão do app: atualize o app e gere de novo.")
    for f in _chain_files(chain):
        if not (workdir / f).is_file():
            raise PackageError(f"Falta o arquivo {f} no pacote: gere de novo pelo app.")
    return {**settings, "dir": str(workdir), "chain": str(workdir / "chain.json")}


def _chain_files(chain: Dict[str, Any]) -> List[str]:
    out = []
    for n in chain.get("stages", []):
        stages = [s for b in n["branches"] for s in b["stages"]] if n.get("type") == "split" else [n]
        out += [s["file"] for s in stages]
    return out


# ------------------------------------------------------------ sinal de treino


def _md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_v3(dest: Path, url: str = V3_URL) -> Path:
    """Baixa o v3_0_0.wav oficial (uma vez) e confere o md5."""
    dest = Path(dest)
    if dest.is_file() and _md5(dest) == V3_MD5:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "nam-chain-capture"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    if _md5(dest) != V3_MD5:
        dest.unlink(missing_ok=True)
        raise RuntimeError("O download do sinal de treino veio corrompido. Rode esta etapa de novo.")
    return dest


# -------------------------------------------------------------------- treino


def train_command(job: Dict[str, Any], v3: Path, out: Path, extra: Optional[List[str]] = None) -> List[str]:
    cmd = [sys.executable, "-m", "chaincap", "run", job["chain"], "--input", str(v3),
           "--out", str(out), "--arch", job.get("arch", "a2"),
           "--epochs", str(int(job.get("epochs", 100)))]
    if not job.get("warm_start", True):
        cmd.append("--no-warm-start")
    if not job.get("auto_stop", True):
        cmd.append("--no-auto-stop")
    return cmd + list(extra or [])


def progress_text(p: Optional[Dict[str, Any]], elapsed: float) -> str:
    mins = f"{int(elapsed // 60)} min {int(elapsed % 60):02d} s"
    if not p or not p.get("max_epochs"):
        return f"Preparando a cadeia... ({mins})"
    done = len(p.get("esr_history") or [])
    txt = f"Época {done} de {p['max_epochs']}"
    if p.get("best_val_loss") is not None:
        txt += f" · melhor ESR {p['best_val_loss']:.4f}"
    if p.get("warm_start"):
        txt += f" · começou de {p['warm_start'].removesuffix('.nam')}"
    return f"{txt} ({mins})"


def _show(handle, text: str, pct: float, done: bool = False) -> None:
    color = "#2e9d5b" if done else "#e3b23c"
    block = (
        f'<div style="font:15px sans-serif;max-width:640px">'
        f'<div style="margin-bottom:6px">{html.escape(text)}</div>'
        f'<div style="background:#333;border-radius:6px;height:14px;overflow:hidden">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{color}"></div></div></div>'
    )
    if handle is not None:
        from IPython.display import HTML

        handle.update(HTML(block))


def train(job: Dict[str, Any], v3: Path, out: Path, extra: Optional[List[str]] = None,
          cwd: Optional[Path] = None, on_update: Optional[Callable[[str, float], None]] = None,
          poll: float = 5.0) -> Dict[str, Any]:
    """Treina com barra de progresso no notebook. Devolve o report.json."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    handle = None
    try:
        from IPython.display import HTML, display

        handle = display(HTML(""), display_id=True)
    except Exception:  # fora do notebook (testes)
        pass
    log = Path(str(out) + ".log")
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.Popen(train_command(job, v3, out, extra), cwd=cwd,
                                stdout=lf, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            p = None
            try:
                p = json.loads((out / "progress.json").read_text())
            except (OSError, json.JSONDecodeError):
                pass
            text = progress_text(p, time.time() - t0)
            pct = 100 * len((p or {}).get("esr_history") or []) / max(1, (p or {}).get("max_epochs") or 1)
            _show(handle, text, pct)
            if on_update:
                on_update(text, pct)
            time.sleep(poll)
    report_path = out / "report.json"
    if not report_path.is_file():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"O treino falhou. Últimas linhas do log:\n{tail}")
    report = json.loads(report_path.read_text())
    stop = f" Parou antes: {report['stop_reason']}." if report.get("stop_reason") else ""
    _show(handle, f"Pronto em {int((time.time() - t0) // 60)} min: ESR {report['esr']:.4f} "
                  f"({report['esr_verdict']}), {report['epochs_run']} épocas.{stop}", 100, done=True)
    return report


# ------------------------------------------------------------------- entrega


def result_file(job: Dict[str, Any], out: Path) -> Path:
    """Copia o `.nam` para o nome do treino (ex.: "Meu rig.nam")."""
    src = Path(out) / "chain.nam"
    dest = Path(out) / f"{safe_filename(job.get('name', ''))}.nam"
    shutil.copyfile(src, dest)
    return dest
