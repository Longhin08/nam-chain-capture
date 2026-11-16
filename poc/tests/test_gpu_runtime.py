import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import gpu_runtime as G

POC = Path(__file__).resolve().parents[1]

INDEX_HTML = """<html><body>
<a href="/whl/cu126/torch-2.9.0%2Bcu126-cp312-cp312-win_amd64.whl#sha256={a}">x</a>
<a href="/whl/cu126/torch-{v}%2Bcu126-cp311-cp311-win_amd64.whl#sha256={b}">x</a>
<a href="/whl/cu126/torch-{v}%2Bcu126-cp312-cp312-win_amd64.whl#sha256={c}">x</a>
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


def test_find_wheel_picks_exact_version_python_and_platform(monkeypatch):
    v = "2.15.0"
    calls = []

    def fake_open(url, timeout=0):
        calls.append(url)
        if "cu126" in url:
            return _Resp(INDEX_HTML.format(v=v, a="a" * 64, b="b" * 64, c="c" * 64))
        return _Resp("<html></html>")

    monkeypatch.setattr(G.urllib.request, "urlopen", fake_open)
    url, sha, tag = G.find_wheel(v, "cp312", "win_amd64")
    assert tag == "cu126"
    assert sha == "c" * 64
    assert url == f"https://download.pytorch.org/whl/cu126/torch-{v}%2Bcu126-cp312-cp312-win_amd64.whl"
    # procurou dos tags mais novos para os mais velhos até achar
    assert calls[0].endswith("/cu130/torch/")


def test_find_wheel_raises_when_missing(monkeypatch):
    monkeypatch.setattr(G.urllib.request, "urlopen", lambda url, timeout=0: _Resp("<html></html>"))
    with pytest.raises(LookupError):
        G.find_wheel("9.9.9", "cp312", "win_amd64")


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
