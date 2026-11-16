"""Monta a versão portátil do Windows, sem nenhum .exe próprio.

    python packaging/make_portable.py --version 0.1.1

Roda no Windows (CI), com um Python 3.x x64 da mesma versão do embeddable.

O Controle Inteligente de Aplicativos do Windows 11 barra executáveis sem
assinatura e sem reputação (como o .exe do PyInstaller e o instalador). Aqui
só rodam binários já conhecidos:
  - python.exe / pythonw.exe do Python "embeddable" oficial (assinados pela PSF);
  - bibliotecas instaladas pelo pip sem modificação (mesmos arquivos de sempre);
  - o código do app como .py, executado pelo pythonw.exe.
A interface abre no navegador padrão (sem pywebview/.NET).

Saída: dist/NAM-Chain-Capture-<versão>-Windows-portable.zip
"""

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

POC = Path(__file__).resolve().parents[1]
NAME = "NAM Chain Capture"

# Só o que o app usa em tempo de execução (sem pytest, pyinstaller, pywebview).
PACKAGES = [
    "neural-amp-modeler==0.13.0",
    "scipy",
    "numpy",
    "fastapi",
    "uvicorn",
    "python-multipart",
]

LAUNCHER = r"""@echo off
rem Abre o NAM Chain Capture (janela do Edge em modo aplicativo) com o Python
rem oficial desta pasta. Na primeira vez, cria atalhos no Menu Iniciar e na
rem Area de trabalho.
start "" "%~dp0python\pythonw.exe" "%~dp0app\desktop_main.py" --browser --portable
"""

README = """NAM Chain Capture {version} - versão portátil para Windows
============================================================

1. Extraia esta pasta inteira para um lugar fixo (por exemplo, Documentos).
   Se mudar a pasta de lugar depois, abra pelo .cmd de novo para refazer os atalhos.
2. Dê dois cliques em "Abrir NAM Chain Capture.cmd".
   - O app abre na própria janela (Microsoft Edge em modo aplicativo, sem abas
     nem barra de endereço). A primeira abertura demora um pouco.
   - Na primeira vez, ele cria os atalhos "NAM Chain Capture" no Menu Iniciar
     e na Área de trabalho. Depois disso, abra por eles.
3. Para fechar, feche a janela. O app se encerra sozinho em cerca de 1 minuto
   (se houver um treino rodando, ele espera o treino terminar).

Seus arquivos (capturas, IRs, treinos) ficam em %APPDATA%\\NAM Chain Capture.

Por que não tem .exe: esta versão roda só com o Python oficial (assinado pela
Python Software Foundation) e bibliotecas originais, para passar pelo Controle
Inteligente de Aplicativos do Windows 11 enquanto o app não tem assinatura
digital própria.

Se o atalho .cmd for bloqueado, crie um atalho manualmente:
  Destino:  <esta pasta>\\python\\pythonw.exe "<esta pasta>\\app\\desktop_main.py" --browser --portable
"""


def run(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default=str(POC / "dist"))
    args = ap.parse_args()

    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        sys.exit("Rode no Windows x64 (a versão portátil é só para Windows).")

    pyver = platform.python_version()  # ex.: 3.12.10
    tag = f"{sys.version_info.major}{sys.version_info.minor}"
    root = Path(tempfile.mkdtemp()) / f"{NAME}"
    pydir, appdir = root / "python", root / "app"

    # 1) Python embeddable oficial
    url = f"https://www.python.org/ftp/python/{pyver}/python-{pyver}-embed-amd64.zip"
    zpath = root.parent / "embed.zip"
    print(f"Baixando {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(pydir)
    # caminhos de busca: stdlib, site-packages e o código do app
    (pydir / f"python{tag}._pth").write_text(
        f"python{tag}.zip\n.\nLib\\site-packages\n..\\app\nimport site\n"
    )

    # 2) bibliotecas, sem modificação
    site = pydir / "Lib" / "site-packages"
    run(sys.executable, "-m", "pip", "install", "--no-input", "--target", site,
        "--index-url", "https://download.pytorch.org/whl/cpu", "torch")
    run(sys.executable, "-m", "pip", "install", "--no-input", "--target", site, *PACKAGES)
    # O pip cria .exe de atalho para comandos das bibliotecas (uvicorn.exe,
    # torchrun.exe...). O app não usa nenhum e eles não são assinados: fora.
    shutil.rmtree(site / "bin", ignore_errors=True)
    for exe in site.rglob("*.exe"):
        print(f"removendo {exe.relative_to(root)}")
        exe.unlink()

    # 3) código do app
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    appdir.mkdir(parents=True)
    shutil.copy2(POC / "desktop_main.py", appdir)
    shutil.copytree(POC / "chaincap", appdir / "chaincap", ignore=ignore)
    shutil.copytree(POC / "app", appdir / "app", ignore=ignore)

    # 4) atalho e leia-me
    (root / f"Abrir {NAME}.cmd").write_text(LAUNCHER.replace("\n", "\r\n"), encoding="ascii")
    (root / "LEIA-ME.txt").write_text(README.format(version=args.version).replace("\n", "\r\n"), encoding="utf-8")

    # 5) zip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"NAM-Chain-Capture-{args.version}-Windows-portable"
    archive = shutil.make_archive(str(base), "zip", root.parent, root.name)
    print(f"Pronto: {archive} ({Path(archive).stat().st_size / 2**20:.0f} MB)")
    print(f"PORTABLE_ROOT={root}")


if __name__ == "__main__":
    main()
