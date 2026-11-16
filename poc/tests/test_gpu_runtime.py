import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import gpu_runtime as G

POC = Path(__file__).resolve().parents[1]

INDEX_HTML = """<html><body>
<a href="https://download-r2.pytorch.org/whl/cpu/torch-{v}%2Bcpu-cp312-cp312-win_amd64.whl#sha256={z}">x</a>
<a href="https://download-r2.pytorch.org/whl/cu132/torch-{v}%2Bcu132-cp312-cp312-win_amd64.whl#sha256={a}">x</a>
<a href="https://download-r2.pytorch.org/whl/cu134/torch-{v}%2Bcu134-cp312-cp312-win_amd64.whl#sha256={b}">x</a>
<a href="https://download-r2.pytorch.org/whl/cu134/torch-{v}%2Bcu134-cp311-cp311-win_amd64.whl#sha256={c}">x</a>
<a href="https://download-r2.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256={c}">x</a>
</body></html>"""


class _Resp:
    def __init__(self, body):
        self.body = body.encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def index(monkeypatch):
    v = "2.15.0"
    html = INDEX_HTML.format(v=v, z="0" * 64, a="a" * 64, b="b" * 64, c="c" * 64)
    monkeypatch.setattr(G.urllib.request, "urlopen", lambda url, timeout=0: _Resp(html))
    return v


def test_find_wheel_picks_newest_cuda_the_driver_supports(index):
    url, sha, tag = G.find_wheel(index, "cp312", "win_amd64", max_cuda=(13, 2))
    assert tag == "cu132" and sha == "a" * 64
    assert url.endswith("/cu132/torch-2.15.0%2Bcu132-cp312-cp312-win_amd64.whl")
    # driver mais novo: pega a variante mais nova
    assert G.find_wheel(index, "cp312", "win_amd64", max_cuda=(13, 4))[2] == "cu134"
    assert G.find_wheel(index, "cp312", "win_amd64")[2] == "cu134"  # driver desconhecido


def test_find_wheel_explains_old_driver(index):
    with pytest.raises(G.DriverTooOld, match="CUDA 13.2"):
        G.find_wheel(index, "cp312", "win_amd64", max_cuda=(12, 8))


def test_find_wheel_raises_when_version_missing(index):
    with pytest.raises(LookupError):
        G.find_wheel("9.9.9", "cp312", "win_amd64")


def test_cuda_tag_version():
    assert G.cuda_tag_version("cu132") == (13, 2)
    assert G.cuda_tag_version("cu118") == (11, 8)
    assert G.cuda_tag_version("cu92") == (9, 2)


def test_marker_ignored_when_torch_version_changes(tmp_path):
    rt = tmp_path / "gpu-runtime" / "torch-x"
    (rt / "torch").mkdir(parents=True)
    marker = tmp_path / "gpu-runtime" / "runtime.json"
    marker.write_text(json.dumps({"torch_version": G.bundled_torch_version(), "path": str(rt), "device": "RTX"}))
    assert G.read_marker(tmp_path)["device"] == "RTX"
    marker.write_text(json.dumps({"torch_version": "0.0.1", "path": str(rt)}))
    assert G.read_marker(tmp_path) is None


def test_activate_makes_torch_import_come_from_runtime(tmp_path):
    """Num processo novo, `import torch` (e submódulos) carrega do runtime baixado."""
    fake = tmp_path / "rt"
    (fake / "torch" / "cuda").mkdir(parents=True)
    (fake / "torch" / "__init__.py").write_text("ORIGIN = 'runtime'\n")
    (fake / "torch" / "cuda" / "__init__.py").write_text("def is_available():\n    return True\n")
    code = (
        "import os, sys; sys.path.insert(0, %r)\n"
        "from app import gpu_runtime as G\n"
        "print(G.activate())\n"
        "import torch, torch.cuda\n"
        "print(torch.ORIGIN, torch.cuda.is_available())\n"
    ) % str(POC)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=dict(os.environ, CHAINCAP_GPU_RUNTIME=str(fake)),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [str(fake), "runtime", "True"]


def test_extract_rejects_path_traversal(tmp_path):
    import zipfile

    whl = tmp_path / "x.whl"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("torch/../../evil.txt", "x")
    with pytest.raises(ValueError):
        G._extract(whl, tmp_path / "out")
