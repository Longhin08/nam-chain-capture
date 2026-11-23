"""Fluxo do notebook do Colab rodando localmente (sem google.colab nem GPU)."""

import ast
import json
import zipfile
from pathlib import Path

import pytest

from chaincap import colab

from conftest import FIXTURES
from test_train_smoke import _synthetic_input

POC = Path(__file__).resolve().parents[1]


def _package(tmp_path, **settings):
    chain = {"name": "Rig", "arch": "a2",
             "stages": [{"type": "nam", "file": "files/amp.nam"}]}
    s = {"name": "Meu rig/AB", "arch": "a2", "epochs": 1, "warm_start": True,
         "auto_stop": True, "app_version": "0.0.0", **settings}
    return colab.build_package(tmp_path / "p.chaincap", chain,
                               {"amp.nam": FIXTURES / "legacy_nano.nam"}, s)


def test_notebook_flow_trains_and_names_the_result(tmp_path):
    job = colab.open_package(_package(tmp_path), tmp_path / "cadeia")
    inp = _synthetic_input(tmp_path)
    updates = []
    rep = colab.train(job, inp, tmp_path / "resultado", cwd=POC, poll=0.2,
                      extra=["--smoke", "--skip-input-check"],
                      on_update=lambda t, p: updates.append(t))
    assert rep["arch"] == "a2" and rep["epochs_max"] == 1
    out = colab.result_file(job, tmp_path / "resultado")
    assert out.name == "Meu rig_AB.nam"
    assert json.loads(out.read_text())["architecture"] == "SlimmableContainer"
    assert updates and updates[0].startswith(("Preparando", "Época"))


def test_train_command_maps_settings():
    job = {"chain": "c.json", "arch": "lite", "epochs": 30, "warm_start": False, "auto_stop": False}
    cmd = colab.train_command(job, Path("v3.wav"), Path("out"))
    assert cmd[2:5] == ["chaincap", "run", "c.json"]
    assert cmd[cmd.index("--arch") + 1] == "lite" and cmd[cmd.index("--epochs") + 1] == "30"
    assert "--no-warm-start" in cmd and "--no-auto-stop" in cmd


def test_progress_text_is_readable():
    assert colab.progress_text(None, 5).startswith("Preparando")
    t = colab.progress_text({"max_epochs": 100, "esr_history": [0.03, 0.01], "best_val_loss": 0.01,
                             "warm_start": "Amp.nam"}, 125)
    assert t == "Época 2 de 100 · melhor ESR 0.0100 · começou de Amp (2 min 05 s)"


def test_open_package_rejects_bad_files(tmp_path):
    (tmp_path / "x.chaincap").write_text("não é zip")
    with pytest.raises(colab.PackageError, match="não é um pacote"):
        colab.open_package(tmp_path / "x.chaincap", tmp_path / "w1")

    evil = tmp_path / "evil.chaincap"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../fora.txt", "x")
    with pytest.raises(colab.PackageError, match="caminhos"):
        colab.open_package(evil, tmp_path / "w2")
    assert not (tmp_path / "fora.txt").exists()

    pkg = _package(tmp_path)
    with zipfile.ZipFile(pkg) as z, zipfile.ZipFile(tmp_path / "falta.chaincap", "w") as o:
        for n in z.namelist():
            if not n.startswith("files/"):
                o.writestr(n, z.read(n))
    with pytest.raises(colab.PackageError, match="Falta o arquivo"):
        colab.open_package(tmp_path / "falta.chaincap", tmp_path / "w3")


def test_notebook_cells_are_valid_and_use_existing_functions():
    nb = json.loads((POC / "chain_capture_colab.ipynb").read_text())
    assert nb["metadata"]["accelerator"] == "GPU"  # abre pedindo GPU
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    used = set()
    for c in code:
        src = "".join(c["source"])
        assert src.startswith("#@title ") and c["metadata"].get("cellView") == "form"
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "colab":
                used.add(node.attr)
    assert used == {"open_package", "fetch_v3", "train", "result_file"}
    assert all(callable(getattr(colab, f)) for f in used)
