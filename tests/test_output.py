"""Checks on the rendered remix (out/). Heavy ones (Whisper, pYIN) are marked slow."""

import pytest

from remix import checks as C

pytestmark = pytest.mark.output


def _assert(res):
    assert res["ok"], res.get("detail", res)


def test_length_2_40_to_3_00(rendered):
    _assert(C.check_length())


def test_track_format_and_equal_lengths(rendered):
    _assert(C.check_format())


def test_sum_of_tracks_is_master(rendered):
    _assert(C.check_sum_equals_master())


def test_sections_on_b_bar_grid_10ms(rendered):
    _assert(C.check_grid_sections())


def test_a_vocal_onsets_follow_grid(rendered):
    _assert(C.check_vocal_onsets_on_grid())


def test_key_a_vocals_fit_b(rendered):
    _assert(C.check_key())


def test_lufs_dbtp_and_limiter(rendered):
    _assert(C.check_loudness())


def test_no_clips_in_stems(rendered):
    _assert(C.check_no_clips())


def test_intro_lowpass_closed(rendered):
    _assert(C.check_lowpass_intro())


def test_no_simultaneous_syllables(rendered):
    _assert(C.check_vocal_overlap())


def test_drier_hook_than_v1(rendered):
    res = C.check_dry_wet_v1_v2()
    if res.get("skipped"):
        pytest.skip("no v1 stems in work/v1 to compare against")
    _assert(res)


@pytest.mark.slow
def test_autotune_on_b_minor(rendered):
    _assert(C.check_autotune())


@pytest.mark.slow
def test_intelligibility_after_autotune(rendered):
    _assert(C.check_intelligibility())


@pytest.mark.slow
def test_whisper_hears_hook_and_refrain(rendered):
    """Pass condition as documented in checks.check_lyrics_in_remix: the hook is heard the way
    Whisper hears it in the A source ("without you(r) ..."), "fall asleep" at least once.
    The literal plan wording ("don't you feel" everywhere) is reported as plan_literal_met."""
    _assert(C.check_lyrics_in_remix())
