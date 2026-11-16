"""Divisão do sinal (cadeias paralelas somadas no mixer)."""

import numpy as np
import pytest

from chaincap import chain as C
from chaincap.chain import Branch, Split, Stage, apply_ir, load_ir, nam_infer, parse_node, render_chain

from conftest import FIXTURES
from helpers import write_wav

RNG = np.random.default_rng(7)
NANO = FIXTURES / "legacy_nano.nam"
GATED = FIXTURES / "legacy_gated.nam"


def sig(n=24000, amp=0.05):
    return (RNG.standard_normal(n) * amp).astype(np.float32)


def raw(res):
    return res.y / res.clip_gain


@pytest.fixture
def ir(tmp_path):
    h = np.zeros(32, np.float32)
    h[:3] = [1.0, 0.4, 0.1]
    return write_wav(tmp_path / "cab.wav", h)


def test_split_sums_branches_with_level_and_phase(ir):
    x = sig()
    m, _ = C.load_nam_model(NANO)
    h, _ = load_ir(ir)
    a = nam_infer(m, x)
    b = apply_ir(x, h)
    res = render_chain(x, [Split(branches=[
        Branch(stages=[Stage("nam", NANO)], level_db=-6),
        Branch(stages=[Stage("ir", ir)], level_db=0, invert=True),
    ])])
    np.testing.assert_allclose(raw(res), C.db_to_gain(-6) * a - b, atol=1e-5)


def test_pre_and_post_stages_wrap_the_split(ir):
    x = sig()
    pre = Stage("nam", NANO, input_gain_db=3)
    post = Stage("ir", ir)
    res = render_chain(x, [pre, Split(branches=[Branch(), Branch(stages=[Stage("nam", GATED)])]), post])
    m1, _ = C.load_nam_model(NANO)
    m2, _ = C.load_nam_model(GATED)
    h, _ = load_ir(ir)
    y = nam_infer(m1, x * C.db_to_gain(3))
    y = y + nam_infer(m2, y)  # cadeia A vazia = sinal limpo
    np.testing.assert_allclose(raw(res), apply_ir(y, h), atol=1e-5)


def test_muted_branch_is_skipped():
    x = sig()
    m, _ = C.load_nam_model(NANO)
    res = render_chain(x, [Split(branches=[
        Branch(stages=[Stage("nam", NANO)]),
        Branch(stages=[Stage("nam", GATED)], enabled=False),
    ])])
    np.testing.assert_allclose(raw(res), nam_infer(m, x), atol=1e-5)
    assert res.stages_metadata[0]["branches"][1]["enabled"] is False


def test_all_muted_and_two_splits_are_rejected():
    x = sig(1000)
    with pytest.raises(ValueError, match="mudo"):
        render_chain(x, [Split(branches=[Branch(enabled=False), Branch(enabled=False)])])
    two = Split(branches=[Branch(), Branch()])
    with pytest.raises(ValueError, match="Só uma divisão"):
        render_chain(x, [two, two])


def test_parse_node_reads_split_and_rejects_nesting(tmp_path):
    node = parse_node({"type": "split", "branches": [
        {"stages": [{"type": "nam", "file": "a.nam"}], "level_db": -3, "invert": True},
        {"stages": []},
    ]}, base_dir=tmp_path)
    assert isinstance(node, Split)
    assert node.branches[0].stages[0].file == tmp_path / "a.nam"
    assert node.branches[0].invert and node.branches[0].level_db == -3
    with pytest.raises(ValueError, match="dentro de divisões"):
        parse_node({"type": "split", "branches": [{"stages": [{"type": "split", "branches": []}]}, {}]})
    with pytest.raises(ValueError, match="2 a 4"):
        parse_node({"type": "split", "branches": [{}]})


def test_calibration_comes_from_first_nam_in_first_live_branch():
    res = render_chain(sig(4000), [Split(branches=[
        Branch(stages=[Stage("nam", NANO, input_gain_db=2)]),  # input_level_dbu=12
        Branch(stages=[Stage("nam", GATED)]),
    ])])
    assert res.input_level_dbu == pytest.approx(10.0)
