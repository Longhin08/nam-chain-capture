import numpy as np
import pytest
import torch

from chaincap import chain as C
from chaincap.chain import Stage, apply_ir, load_ir, nam_infer, render_chain

from conftest import FIXTURES
from helpers import export_nam, random_wavenet, write_wav

RNG = np.random.default_rng(42)


def sig(n=48000, amp=0.2):
    return (RNG.standard_normal(n) * amp).astype(np.float32)


def test_chunked_inference_equals_single_pass(monkeypatch):
    m = random_wavenet("nano")
    x = sig(30000)
    with torch.no_grad():
        ref = m(torch.from_numpy(x), pad_start=True).numpy()
    monkeypatch.setattr(C, "_CHUNK", 7000)  # força vários blocos (rf ~ 2k)
    np.testing.assert_allclose(nam_infer(m, x), ref, atol=1e-6)
    assert len(nam_infer(m, x)) == len(x)


def test_dirac_ir_is_identity(tmp_path):
    h = np.zeros(64, np.float32)
    h[0] = 0.5
    p = write_wav(tmp_path / "d.wav", h)
    ir, warns = load_ir(p)
    assert ir[0] == pytest.approx(1.0)  # normalização por energia
    x = sig()
    np.testing.assert_allclose(apply_ir(x, ir), x, atol=1e-6)
    assert warns == []


def test_ir_is_causal_same_length():
    h = np.array([0.0, 1.0], np.float32)  # atraso de 1 amostra
    x = sig(100)
    y = apply_ir(x, h)
    assert len(y) == len(x)
    np.testing.assert_allclose(y[1:], x[:-1], atol=1e-6)
    assert abs(y[0]) < 1e-6


def test_long_ir_warns_and_truncates(tmp_path):
    h = (RNG.standard_normal(48000 // 10) * np.exp(-np.arange(4800) / 800)).astype(np.float32)
    p = write_wav(tmp_path / "cab.wav", h)  # 100 ms
    _, warns = load_ir(p)
    assert any("100 ms" in w for w in warns)
    ir, warns = load_ir(p, max_ms=20)
    assert len(ir) == 960
    assert any("truncado" in w for w in warns)
    assert not any("cauda" in w for w in warns)


def test_ir_resampled_to_48k(tmp_path):
    h = np.zeros(441, np.float32)
    h[0] = 1
    p = write_wav(tmp_path / "ir441.wav", h, sr=44100)
    ir, warns = load_ir(p)
    assert len(ir) == 480
    assert any("reamostrado" in w for w in warns)


def test_render_chain_applies_gains_in_order(tmp_path):
    a = export_nam(random_wavenet("nano", seed=1), tmp_path / "a.nam")
    b = export_nam(random_wavenet("nano", seed=2), tmp_path / "b.nam")
    x = sig(20000)
    ma, _ = C.load_nam_model(a)
    mb, _ = C.load_nam_model(b)
    g = C.db_to_gain
    expected = nam_infer(mb, nam_infer(ma, x * g(6)) * g(-3)) * g(2)

    res = render_chain(x, [
        Stage("nam", a, input_gain_db=6, output_gain_db=-3),
        Stage("nam", b, output_gain_db=2),
    ])
    np.testing.assert_allclose(res.y * (1 / res.clip_gain), expected, atol=1e-5)
    assert [s["file"] for s in res.stages_metadata] == ["a.nam", "b.nam"]

    swapped = render_chain(x, [Stage("nam", b), Stage("nam", a)])
    assert not np.allclose(swapped.y, res.y, atol=1e-3)  # ordem importa


def test_clipping_is_scaled_and_reported(tmp_path):
    h = np.zeros(8, np.float32)
    h[0] = 1
    p = write_wav(tmp_path / "d.wav", h)
    res = render_chain(sig(amp=0.5), [Stage("ir", p, output_gain_db=12)])
    assert np.max(np.abs(res.y)) <= C.PEAK_LIMIT + 1e-6
    assert res.clip_gain < 1
    assert any("clipava" in w for w in res.warnings)


def test_input_calibration_propagates_from_first_nam(tmp_path):
    # legacy_nano.nam tem input_level_dbu = 12
    res = render_chain(sig(8000), [
        Stage("nam", FIXTURES / "legacy_nano.nam", input_gain_db=3),
        Stage("nam", FIXTURES / "legacy_gated.nam", input_gain_db=10),
    ])
    assert res.input_level_dbu == pytest.approx(9.0)


def test_rejects_invalid_nam(tmp_path):
    p = tmp_path / "bad.nam"
    p.write_text('{"foo": 1}')
    with pytest.raises(ValueError, match="válido"):
        render_chain(sig(1000), [Stage("nam", p)])


def test_rejects_wrong_sample_rate(tmp_path):
    m = random_wavenet("nano")
    m.sample_rate = 44100
    p = export_nam(m, tmp_path / "m.nam")
    with pytest.raises(ValueError, match="sample rate"):
        render_chain(sig(1000), [Stage("nam", p)])


def test_stage_from_dict_resolves_relative_paths(tmp_path):
    s = Stage.from_dict({"type": "ir", "file": "cab.wav", "ir_max_ms": 20}, base_dir=tmp_path)
    assert s.file == tmp_path / "cab.wav" and s.ir_max_ms == 20
