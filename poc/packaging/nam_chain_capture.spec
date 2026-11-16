# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: um único executável que abre a janela e também roda o worker.

    cd poc && pyinstaller packaging/nam_chain_capture.spec --noconfirm
Saída: dist/NAM Chain Capture/ (Windows/Linux) ou dist/NAM Chain Capture.app (macOS).
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

POC = Path(SPECPATH).parent
sys.path.insert(0, str(POC))
from app.version import VERSION  # noqa: E402

APP = "NAM Chain Capture"

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("nam")
    + collect_submodules("pytorch_lightning")
    + collect_submodules("lightning_fabric")
    + collect_submodules("chaincap")
    + collect_submodules("app")
    + ["webview"]
)
datas = [(str(POC / "app" / "static"), "app/static")]
datas += collect_data_files("nam")
datas += collect_data_files("lightning_fabric")
datas += collect_data_files("pytorch_lightning")
# Lightning/torchmetrics checam versões de dependências via importlib.metadata.
datas += copy_metadata("neural-amp-modeler", recursive=True)
datas += copy_metadata("fastapi", recursive=True)

a = Analysis(
    [str(POC / "desktop_main.py")],
    pathex=[str(POC)],
    hiddenimports=hidden,
    datas=datas,
    excludes=[
        "tkinter", "_tkinter", "IPython", "jupyter", "notebook", "tensorboard",
        "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_qtagg",
        "PyQt5", "PyQt6", "PySide2", "PySide6",
        # Dependências do librosa/nam que o pipeline nunca carrega (~200 MB).
        # O --self-test do CI pega se isso mudar.
        "numba", "llvmlite", "sklearn", "pandas",
    ],
    noarchive=False,
    # O Lightning define __version__ checando se __version__.py existe em disco;
    # dentro do PYZ ele não existe e o checkpoint quebra. Mantém os .py também.
    module_collection_mode={
        "pytorch_lightning": "pyz+py",
        "lightning_fabric": "pyz+py",
    },
)
pyz = PYZ(a.pure)

icon = str(POC / "packaging" / ("icon.ico" if sys.platform == "win32" else "icon.icns"))
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    console=False,
    icon=icon,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP}.app",
        icon=icon,
        bundle_identifier="com.namchaincapture.app",
        version=VERSION,
        info_plist={
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
