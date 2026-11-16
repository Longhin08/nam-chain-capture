"""Ponto de entrada do app desktop (e do executável do PyInstaller).

    NAM Chain Capture                  abre a janela do app
    NAM Chain Capture --browser        abre no navegador padrão em vez da janela
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
        self.ws = workspace
        self.window = None

    def save_job_file(self, job_id: str, name: str):
        import webview

        if not re.fullmatch(r"[0-9a-f]{12}", job_id or "") or name != Path(name).name:
            return {"error": "Arquivo inválido"}
        src = self.ws / "jobs" / job_id / "out" / name
        if not src.is_file():
            return {"error": "Arquivo não encontrado"}
        res = self.window.create_file_dialog(webview.FileDialog.SAVE, save_filename=name)
        if not res:
            return {"canceled": True}
        dest = res if isinstance(res, str) else res[0]
        shutil.copy2(src, dest)
        return {"saved": dest}


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
        api.window = webview.create_window(
            APP_NAME, url, js_api=api, width=1440, height=920, min_size=(960, 640),
            background_color="#1d2737",
        )
        webview.start(private_mode=False, storage_path=str(ws / "webview"))
    except Exception as e:
        import webbrowser

        if str(e) != "--browser":
            print(f"Janela nativa indisponível ({e}); abrindo no navegador.", file=sys.stderr)
        webbrowser.open(url)
        print(f"{APP_NAME} rodando em {url}\nFeche esta janela (ou Ctrl+C) para sair.")
        try:
            while t.is_alive():
                time.sleep(0.5)
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
    run_app(ws, browser="--browser" in args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
