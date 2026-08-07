import numpy as np

from backend.services.manipuri_pronunciation import (
    build_manipuri_pronunciation_plan,
    detect_manipuri_script,
    detect_reference_speech_bounds,
    normalize_manipuri_text,
)


def test_normalize_manipuri_text_collapses_spacing_and_punctuation_variants():
    assert normalize_manipuri_text("  Eigi\u00a0nangbu—nungshi  ") == "Eigi nangbu-nungshi"


def test_detects_meitei_mayek_and_latin_scripts():
    assert detect_manipuri_script("ꯃꯤꯇꯩ") == "meitei_mayek"
    assert detect_manipuri_script("eigi nangbu") == "latin"
    assert detect_manipuri_script("ꯃꯤꯇꯩ eigi") == "mixed"


def test_meitei_vowel_sign_stays_attached_to_base_unit():
    plan = build_manipuri_pronunciation_plan("ꯃꯤ ꯇꯩ")

    assert plan["script"] == "meitei_mayek"
    assert [unit["text"] for unit in plan["units"]] == ["ꯃꯤ", "ꯇꯩ"]
    assert "MEETEI MAYEK VOWEL SIGN INAP" in plan["units"][0]["modifiers"]
    assert "MEETEI MAYEK VOWEL SIGN CHEINAP" in plan["units"][1]["modifiers"]


def test_lum_iyek_is_preserved_as_pronunciation_marker():
    plan = build_manipuri_pronunciation_plan("ꯃ꯬")

    assert plan["markers"]["lum_iyek_count"] == 1
    assert plan["units"][0]["has_lum_iyek"] is True
    assert "MEETEI MAYEK LUM IYEK" in plan["units"][0]["unicode_names"]


def test_romanized_text_warns_native_reading_is_authority():
    plan = build_manipuri_pronunciation_plan("Eigi nangbu nungshi")

    assert plan["unit_sequence"][:4] == ["e", "i", "g", "i"]
    assert plan["representation"]["native_reading_is_pronunciation_authority"] is True
    assert any("Romanized Manipuri" in warning for warning in plan["warnings"])


def test_reference_speech_bounds_finds_active_region():
    sample_rate = 1000
    audio = np.zeros(2000, dtype=np.float32)
    audio[500:1500] = 0.25

    bounds = detect_reference_speech_bounds(
        audio,
        sample_rate,
        frame_ms=20,
        padding_ms=0,
    )

    assert bounds["status"] == "speech_detected"
    assert bounds["start_sec"] == 0.5
    assert bounds["end_sec"] == 1.5
    assert bounds["active_duration_sec"] == 1.0


def test_reference_speech_bounds_handles_silence():
    audio = np.zeros(16000, dtype=np.float32)

    bounds = detect_reference_speech_bounds(audio, 16000)

    assert bounds["status"] == "no_speech_detected"
    assert bounds["active_duration_sec"] == 0.0
