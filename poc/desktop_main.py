"""Ponto de entrada do app desktop (e do executável do PyInstaller).

    NAM Chain Capture                  abre a janela do app
    NAM Chain Capture --browser        sem pywebview: no Windows abre uma janela do Edge
                                       em modo aplicativo; nos outros, o navegador padrão
    … --portable                       (versão portátil) cria atalhos no Menu Iniciar
                                       e na Área de trabalho na primeira execução
    … --make-shortcuts                 só cria os atalhos e sai
    NAM Chain Capture --worker run …   (interno) treino de um job
    NAM Chain Capture --gpu-check      (interno) testa o PyTorch CUDA baixado
    NAM Chain Capture --serve [PORTA]  só o servidor, sem janela (testes)
    NAM Chain Capture --self-test      roda o pipeline inteiro num caso sintético

O mesmo executável faz tudo: o servidor chama a si mesmo com `--worker` para
treinar num processo separado (dá para cancelar e não trava a interface).
"""

import json
import multiprocessing
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

APP_NAME = "NAM Chain Capture"


def _ensure_streams() -> None:
    """Executável 'windowed' pode nascer sem stdout/stderr; nunca deixar print quebrar."""
    log = os.environ.get("CHAINCAP_LOG")
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(log, "a", buffering=1, encoding="utf-8") if log else open(os.devnull, "w"))


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "nam-chain-capture"


def _setup_env() -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")  # o nam importa pyplot; sem GUI
    ws = Path(os.environ.get("CHAINCAP_WORKSPACE") or data_dir())
    ws.mkdir(parents=True, exist_ok=True)
    os.environ["CHAINCAP_WORKSPACE"] = str(ws)
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    return ws


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DesktopApi:
    """Funções nativas expostas ao JavaScript (window.pywebview.api)."""

    def __init__(self, workspace: Path):
        # Atributos com "_" não são expostos ao JavaScript pelo pywebview.
        self._ws = workspace
        self._window = None

    def save_job_file(self, job_id: str, name: str):
        import webview

        if not re.fullmatch(r"[0-9a-f]{12}", job_id or "") or name != Path(name).name:
            return {"error": "Arquivo inválido"}
        src = self._ws / "jobs" / job_id / "out" / name
        if not src.is_file():
            return {"error": "Arquivo não encontrado"}
        res = self._window.create_file_dialog(webview.FileDialog.SAVE, save_filename=name)
        if not res:
            return {"canceled": True}
        dest = res if isinstance(res, str) else res[0]
        shutil.copy2(src, dest)
        return {"saved": dest}


# ------------------------------------------------- Windows: janela e atalhos

def find_edge():
    """Caminho do msedge.exe (vem instalado no Windows 10/11)."""
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            p = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if p.is_file():
                return p
    return shutil.which("msedge")


def open_app_window(url: str, ws: Path) -> bool:
    """Abre a interface numa janela do Edge em modo aplicativo (sem abas nem barra
    de endereço). O Edge é assinado pela Microsoft, então passa pelo Controle
    Inteligente de Aplicativos. Retorna False se não der (aí usa o navegador)."""
    import subprocess

    edge = find_edge() if sys.platform == "win32" else None
    if not edge:
        return False
    try:
        subprocess.Popen(
            [str(edge), f"--app={url}", f"--user-data-dir={ws / 'janela'}",
             "--window-size=1440,920", "--no-first-run", "--no-default-browser-check"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except OSError:
        return False


def portable_root() -> Path:
    # <raiz>/app/desktop_main.py  e  <raiz>/python/pythonw.exe
    return Path(__file__).resolve().parent.parent


_SHORTCUT_PS = r"""
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in $env:CC_DIRS.Split('|')) {
  if (-not $dir) { continue }
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $lnk = $shell.CreateShortcut((Join-Path $dir ($env:CC_NAME + '.lnk')))
  $lnk.TargetPath = $env:CC_TARGET
  $lnk.Arguments = $env:CC_ARGS
  $lnk.WorkingDirectory = $env:CC_WORKDIR
  $lnk.IconLocation = $env:CC_ICON
  $lnk.Description = 'Monte a cadeia de capturas NAM e treine uma captura única'
  $lnk.Save()
}
"""


def create_shortcuts(ws: Path, force: bool = False) -> list:
    """Atalhos no Menu Iniciar e na Área de trabalho, apontando para o pythonw.exe
    oficial (nenhum .exe próprio). Só na versão portátil do Windows."""
    import subprocess

    if sys.platform != "win32":
        return []
    root = portable_root()
    pythonw = root / "python" / "pythonw.exe"
    if not pythonw.is_file():
        return []
    marker = ws / ".atalhos"
    if not force and marker.is_file() and marker.read_text(encoding="utf-8") == str(root):
        return []  # já criados para esta pasta
    override = os.environ.get("CHAINCAP_SHORTCUT_DIRS")  # testes
    if override:
        dirs = override.split("|")
    else:
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        dirs = [
            str(appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"),
            str(Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop"),
        ]
    env = dict(
        os.environ,
        CC_DIRS="|".join(dirs),
        CC_NAME=APP_NAME,
        CC_TARGET=str(pythonw),
        CC_ARGS=f'"{root / "app" / "desktop_main.py"}" --browser --portable',
        CC_WORKDIR=str(root),
        CC_ICON=str(root / "app" / "app" / "assets" / "icon.ico"),
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _SHORTCUT_PS],
        env=env, check=True, capture_output=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    marker.write_text(str(root), encoding="utf-8")
    return [str(Path(d) / f"{APP_NAME}.lnk") for d in dirs]


# No modo navegador, a página manda /api/ping a cada 10 s. Sem ping por
# AUTOSTOP_S (ou sem nenhum ping depois de GRACE_S), o app se encerra, a não
# ser que haja treino rodando ou na fila.
AUTOSTOP_S = 45
GRACE_S = 180


def should_autostop(last_ping, started, now, busy) -> bool:
    if busy:
        return False
    if last_ping is None:
        return now - started > GRACE_S
    return now - last_ping > AUTOSTOP_S


def run_app(ws: Path, browser: bool) -> None:
    os.environ["CHAINCAP_DESKTOP"] = "1"
    os.environ["CHAINCAP_AUTO_V3"] = "1"
    token = secrets.token_urlsafe(24)
    os.environ["CHAINCAP_TOKEN"] = token

    import uvicorn

    from app import server

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}/?token={token}"

    try:
        if browser:
            raise RuntimeError("--browser")
        import webview

        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
        api = DesktopApi(ws)
        api._window = webview.create_window(
            APP_NAME, url, js_api=api, width=1440, height=920, min_size=(960, 640),
            background_color="#1d2737",
        )
        webview.start(private_mode=False, storage_path=str(ws / "webview"))
    except Exception as e:
        import webbrowser

        if str(e) != "--browser":
            print(f"Janela nativa indisponível ({e}); abrindo no navegador.", file=sys.stderr)
        if not open_app_window(url, ws):
            webbrowser.open(url)
        print(f"{APP_NAME} rodando em {url}\nFeche a aba do navegador (ou Ctrl+C) para sair.")
        started = time.time()
        try:
            while t.is_alive():
                time.sleep(1)
                if should_autostop(server.LAST_PING["t"], started, time.time(), server.jobs_busy()):
                    break
        except KeyboardInterrupt:
            pass
    finally:
        server.shutdown_jobs()
        srv.should_exit = True
        t.join(timeout=5)


def serve(port: int) -> None:
    import uvicorn

    from app import server

    print(f"http://127.0.0.1:{port}/?token={os.environ.get('CHAINCAP_TOKEN', '')}", flush=True)
    try:
        uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        server.shutdown_jobs()


def self_test() -> int:
    """Smoke test do build: gera captura + IR, renderiza, treina 1 passo, exporta."""
    import tempfile

    import numpy as np
    import torch

    from chaincap.__main__ import main as cli
    from chaincap.archs import wavenet_config
    from chaincap.chain import write_wav_24
    from nam.models.wavenet import WaveNet

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        torch.manual_seed(0)
        m = WaveNet.init_from_config(wavenet_config("nano"))
        m.sample_rate = 48000
        m.export(td, basename="drive")
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(1_200_000) * 0.1).astype(np.float32)
        x[432_000:480_000] = 0
        x[-480_000:-432_000] = 0
        write_wav_24(td / "input.wav", x)
        ir = np.zeros(32, np.float32)
        ir[:3] = [1.0, 0.5, 0.25]
        write_wav_24(td / "cab.wav", ir)
        (td / "chain.json").write_text(json.dumps({
            "name": "self-test", "arch": "nano",
            "stages": [{"type": "nam", "file": "drive.nam"}, {"type": "ir", "file": "cab.wav"}],
        }))
        rc = cli(["run", str(td / "chain.json"), "--input", str(td / "input.wav"),
                  "--out", str(td / "out"), "--smoke", "--epochs", "1", "--skip-input-check"])
        a1 = json.loads((td / "out" / "chain.a1.nam").read_text())
        ok = rc == 0 and a1["version"] == "0.5.4"
    print("SELF-TEST OK" if ok else "SELF-TEST FALHOU", flush=True)
    return 0 if ok else 1


def gpu_check(ws: Path) -> int:
    from app import gpu_runtime

    path = gpu_runtime.activate(ws)
    import torch

    ok = torch.cuda.is_available()
    print(json.dumps({
        "runtime": path,
        "torch": torch.__version__,
        "cuda": ok,
        "name": torch.cuda.get_device_name(0) if ok else None,
    }))
    return 0


def main() -> int:
    multiprocessing.freeze_support()
    _ensure_streams()
    ws = _setup_env()
    args = sys.argv[1:]
    if args[:1] == ["--worker"]:
        from app import gpu_runtime

        gpu_runtime.activate(ws)
        from chaincap.__main__ import main as cli

        return cli(args[1:])
    if args[:1] == ["--gpu-check"]:
        return gpu_check(ws)
    if args[:1] == ["--self-test"]:
        return self_test()
    if args[:1] == ["--serve"]:
        os.environ.setdefault("CHAINCAP_TOKEN", secrets.token_urlsafe(24))
        serve(int(args[1]) if len(args) > 1 else _free_port())
        return 0
    if "--make-shortcuts" in args:
        for p in create_shortcuts(ws, force=True):
            print(f"atalho: {p}")
        return 0
    if "--portable" in args:
        try:
            create_shortcuts(ws)
        except Exception as e:  # atalho é conveniência: nunca impede o app de abrir
            print(f"Não consegui criar os atalhos: {e}", file=sys.stderr)
    run_app(ws, browser="--browser" in args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
