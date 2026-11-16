import numpy as np
import pytest

from chaincap.cab_ir import CURVE_4X12, LENGTH, make_cab_ir, write_cab_ir
from chaincap.chain import load_ir


def test_cab_ir_follows_target_curve():
    h = make_cab_ir()
    assert len(h) == LENGTH
    H = np.fft.rfft(h, 16384)
    f = np.fft.rfftfreq(16384, 1 / 48000)
    db = lambda fq: 20 * np.log10(np.abs(H[np.argmin(abs(f - fq))]))
    ref = db(1200)
    for fq, g in CURVE_4X12:
        if 60 <= fq <= 12000:
            assert db(fq) - ref == pytest.approx(g, abs=1.0), fq


def test_cab_ir_is_unit_energy_and_tails_off():
    h = make_cab_ir()
    assert np.sum(h.astype(np.float64) ** 2) == pytest.approx(1.0, rel=1e-4)
    assert abs(h[-1]) < 1e-6  # janela de saída: sem corte abrupto


def test_written_cab_loads_without_warnings(tmp_path):
    p = write_cab_ir(tmp_path / "cab.wav")
    h, warns = load_ir(p)
    assert len(h) == LENGTH and warns == []
