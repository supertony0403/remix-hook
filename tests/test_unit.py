"""Unit tests on synthetic signals (no audio sources needed, except rubberband CLI)."""

import numpy as np
import pytest

from remix import config
from remix import grid as G
from remix.arrangement import ARRANGEMENT, BAR, TOTAL_BARS
from remix.edit import cut, zero_cross_near
from remix.master import limiter_gain, master_chain, soft_clip_gain, true_peak_db
from remix import tune as T

SR = config.SR
rng = np.random.default_rng(7)


def test_limiter_keeps_true_peak_under_ceiling():
    x = (rng.standard_normal((SR * 3, 2)) * 0.25).astype(np.float32)
    x[SR:SR + 50] *= 8.0  # hard transient
    g = limiter_gain(x, -1.0)
    assert true_peak_db(x * g[:, None]) <= -1.0 + 0.05
    assert g.max() <= 1.0 and g.min() > 0.0


def test_soft_clip_gain_is_identity_below_threshold_and_bounded():
    x = np.linspace(-2, 2, 4001, dtype=np.float32)[:, None].repeat(2, 1)
    g = soft_clip_gain(x, -6.0, 0.0)
    y = x * g
    small = np.abs(x[:, 0]) < 10 ** (-6 / 20)
    assert np.allclose(g[small], 1.0)
    assert np.abs(y).max() <= 1.0 + 1e-6
    assert np.all(np.diff(y[:, 0]) >= -1e-7)  # monotonic transfer curve


def test_shared_gain_keeps_sum_of_stems_equal_to_master():
    stems = [(rng.standard_normal((SR * 4, 2)) * a).astype(np.float32) for a in (0.2, 0.1, 0.05)]
    mix = np.sum(stems, axis=0)
    res = master_chain(mix, target_lufs=-12.0)
    summed = np.sum([s * res.gain for s in stems], axis=0)
    assert np.abs(summed - res.master).max() < 1e-5
    assert res.dbtp <= config.TARGET_DBTP
    assert abs(res.lufs + 12.0) < 0.2


def test_cut_snaps_to_zero_crossing_and_fades():
    t = np.arange(SR) / SR
    x = np.stack([np.sin(2 * np.pi * 110 * t)] * 2, axis=1).astype(np.float32)
    i = zero_cross_near(x, 12345)
    assert abs(i - 12345) <= int(0.002 * SR)
    assert np.sign(x[i - 1, 0]) != np.sign(x[i, 0]) or x[i, 0] == 0
    seg, shift = cut(x, 0.25, 0.5)
    assert abs(shift) <= int(0.002 * SR)
    s0 = int(round(0.25 * SR)) + shift  # the snapped start itself must be a zero crossing
    assert np.sign(x[s0 - 1, 0]) != np.sign(x[s0, 0]) or abs(x[s0, 0]) < 1e-6
    assert abs(seg[0, 0]) < 1e-3 and abs(seg[-1, 0]) < 1e-3


def test_tempo_and_phase_on_click_track():
    bpm, phase = 150.0, 0.123
    n = SR * 20
    x = np.zeros(n, np.float32)
    for k in range(int((n / SR - phase) / (60 / bpm))):
        s = int((phase + k * 60 / bpm) * SR)
        x[s:s + 200] += np.hanning(200) * np.sign(np.sin(np.arange(200) * 0.9))
    env = G.onset_envelope(x[:, None].repeat(2, 1), SR)
    got_bpm, got_phase, _ = G.fit_tempo(env, SR, 147, 153)
    assert abs(got_bpm - bpm) < 0.05
    d = (got_phase - phase + 0.2) % 0.4 - 0.2
    assert abs(d - G.ONSET_LATENCY_S) < 0.003  # constant estimator latency, see grid.py


def test_correction_snaps_held_note_to_b_minor():
    f0 = np.full(400, 247.0 * 2 ** (0.4 / 12))  # B3 + 40 cent, held
    corr = T.correction_curve(f0, np.ones(400, int), {1: T.TuneStyle(retune_ms=20.0, strength=1.0)})
    out = 69 + 12 * np.log2(f0 / 440) + corr
    assert np.all(np.abs(out[20:] - 59.0) < 0.05)  # B3 = MIDI 59
    assert T.in_scale_fraction(440 * 2 ** ((out - 69) / 12)) > 0.95


def test_unvoiced_frames_are_not_corrected():
    f0 = np.r_[np.full(100, 250.0), np.zeros(100)]
    corr = T.correction_curve(f0, np.ones(200, int), {1: T.RAP})
    assert np.all(np.abs(corr[130:]) < 1e-3)


@pytest.mark.slow
def test_rubberband_freqmap_lead_is_compensated(tmp_path):
    import subprocess
    import soundfile as sf

    t = np.arange(4 * SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    sf.write(tmp_path / "s.wav", np.stack([x, x], 1), SR, subtype="FLOAT")
    step = 2.0
    frames = np.arange(0, 4 * SR, 240)
    lines = [f"{int(round(fr + T.RB_FREQMAP_LEAD_S * SR))} {1.0 if fr < step * SR else 2 ** (2 / 12):.6f}"
             for fr in frames]
    (tmp_path / "s.map").write_text("0 1.0\n" + "\n".join(lines) + "\n")
    subprocess.run(["rubberband", "-q", "-3", "-F", "--freqmap", str(tmp_path / "s.map"), "-t", "1.0",
                    str(tmp_path / "s.wav"), str(tmp_path / "o.wav")], check=True)
    y, _ = sf.read(tmp_path / "o.wav")
    y = y.mean(1)
    zc = np.nonzero((y[:-1] < 0) & (y[1:] >= 0))[0]
    f = SR / np.diff(zc)
    tt = zc[1:] / SR
    idx = np.nonzero((tt > step - 0.3) & (f > 233.0))[0][0]
    assert abs(tt[idx] - step) < 0.015


def test_arrangement_is_contiguous_and_on_bars():
    secs = ARRANGEMENT.sections
    assert secs[0].start == 0
    for a, b in zip(secs, secs[1:]):
        assert a.end == b.start
    assert secs[-1].end == TOTAL_BARS
    for c in ARRANGEMENT.clips:
        assert c.track in config.TRACK_NAMES
        assert c.end > c.start
        assert -0.5 <= c.remix_start and c.remix_end <= TOTAL_BARS + ARRANGEMENT.tail_bars
    total_s = (TOTAL_BARS + ARRANGEMENT.tail_bars) * BAR
    assert 160.0 <= total_s <= 180.0
