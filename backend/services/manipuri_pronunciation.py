"""Manipuri / Meiteilon pronunciation preparation utilities.

V1 intentionally separates *orthographic pronunciation units* from true
phonemes.  The native reading is the pronunciation authority; text units are
stable alignment keys that a later acoustic aligner can attach timestamps and
pronunciation embeddings to.

This avoids pretending that a generic Latin/English G2P is accurate for
Manipuri while still giving the singing pipeline a deterministic text layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Literal

import numpy as np


ScriptKind = Literal["meitei_mayek", "latin", "mixed", "unknown"]

# Unicode blocks:
#   Meetei Mayek Extensions: U+AAE0..U+AAFF
#   Meetei Mayek:            U+ABC0..U+ABFF
MEETEI_EXT_START = 0xAAE0
MEETEI_EXT_END = 0xAAFF
MEETEI_START = 0xABC0
MEETEI_END = 0xABFF

LUM_IYEK = "\uABEC"
APUN_IYEK = "\uABED"
MEETEI_VIRAMA = "\uAAF6"

_APOSTROPHES_RE = re.compile(r"[\u2018\u2019\u02BC\uFF07]")
_DASHES_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PronunciationUnit:
    """One deterministic orthographic unit used as an alignment key."""

    index: int
    word_index: int
    text: str
    script: ScriptKind
    codepoints: list[str]
    unicode_names: list[str]
    modifiers: list[str]
    has_lum_iyek: bool
    has_apun_iyek: bool
    has_virama: bool
    alignment_key: str


@dataclass(frozen=True)
class PronunciationWord:
    index: int
    text: str
    start_char: int
    end_char: int
    unit_indexes: list[int]


def _is_meitei_char(char: str) -> bool:
    cp = ord(char)
    return (MEETEI_EXT_START <= cp <= MEETEI_EXT_END) or (MEETEI_START <= cp <= MEETEI_END)


def _is_latin_letter(char: str) -> bool:
    if not char.isalpha():
        return False
    return "LATIN" in unicodedata.name(char, "")


def _char_script(char: str) -> ScriptKind:
    if _is_meitei_char(char):
        return "meitei_mayek"
    if _is_latin_letter(char):
        return "latin"
    return "unknown"


def normalize_manipuri_text(text: str) -> str:
    """Normalize user text without transliterating or changing pronunciation."""

    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\u00A0", " ")
    normalized = _APOSTROPHES_RE.sub("'", normalized)
    normalized = _DASHES_RE.sub("-", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def detect_manipuri_script(text: str) -> ScriptKind:
    """Detect Meitei Mayek, Latin, mixed, or unknown text."""

    normalized = normalize_manipuri_text(text)
    meitei = 0
    latin = 0

    for char in normalized:
        if _is_meitei_char(char):
            # Ignore punctuation/digits when deciding that the text is Meitei.
            category = unicodedata.category(char)
            if category.startswith(("L", "M")):
                meitei += 1
        elif _is_latin_letter(char):
            latin += 1

    if meitei and latin:
        return "mixed"
    if meitei:
        return "meitei_mayek"
    if latin:
        return "latin"
    return "unknown"


def _is_word_char(char: str) -> bool:
    if _is_meitei_char(char):
        return not unicodedata.category(char).startswith("P")
    if _is_latin_letter(char) or char.isdigit():
        return True
    if unicodedata.category(char).startswith("M"):
        return True
    return char in {"'", "-"}


def _iter_words(text: str) -> list[tuple[str, int, int]]:
    words: list[tuple[str, int, int]] = []
    start: int | None = None

    for i, char in enumerate(text):
        if _is_word_char(char):
            if start is None:
                start = i
        elif start is not None:
            token = text[start:i].strip("'-")
            if token:
                token_start = start
                while token_start < i and text[token_start] in "'-":
                    token_start += 1
                token_end = i
                while token_end > token_start and text[token_end - 1] in "'-":
                    token_end -= 1
                words.append((text[token_start:token_end], token_start, token_end))
            start = None

    if start is not None:
        end = len(text)
        token = text[start:end].strip("'-")
        if token:
            token_start = start
            while token_start < end and text[token_start] in "'-":
                token_start += 1
            token_end = end
            while token_end > token_start and text[token_end - 1] in "'-":
                token_end -= 1
            words.append((text[token_start:token_end], token_start, token_end))

    return words


def _grapheme_clusters(word: str) -> list[str]:
    """Group base characters with following Unicode marks.

    Meitei vowel signs, LUM IYEK, APUN IYEK and the extension VIRAMA are
    Unicode marks, so keeping them with their base gives us stable units for
    later forced alignment and G2P work.
    """

    clusters: list[str] = []
    current = ""

    for char in word:
        category = unicodedata.category(char)
        is_mark = category.startswith("M")

        if not current:
            current = char
        elif is_mark:
            current += char
        else:
            clusters.append(current)
            current = char

    if current:
        clusters.append(current)

    return clusters


def _cluster_script(cluster: str) -> ScriptKind:
    scripts = {_char_script(c) for c in cluster if _char_script(c) != "unknown"}
    if scripts == {"meitei_mayek"}:
        return "meitei_mayek"
    if scripts == {"latin"}:
        return "latin"
    if len(scripts) > 1:
        return "mixed"
    return "unknown"


def _alignment_key(cluster: str, script: ScriptKind) -> str:
    # Case is not pronunciation-significant for Romanized guide text.
    if script == "latin":
        return cluster.casefold()
    return cluster


def build_manipuri_pronunciation_plan(text: str) -> dict:
    """Create V1's deterministic text/pronunciation representation."""

    normalized = normalize_manipuri_text(text)
    script = detect_manipuri_script(normalized)
    words_raw = _iter_words(normalized)

    units: list[PronunciationUnit] = []
    words: list[PronunciationWord] = []

    for word_index, (word_text, start_char, end_char) in enumerate(words_raw):
        unit_indexes: list[int] = []

        for cluster in _grapheme_clusters(word_text):
            unit_index = len(units)
            cluster_script = _cluster_script(cluster)
            names = [unicodedata.name(c, f"U+{ord(c):04X}") for c in cluster]
            modifiers = [
                unicodedata.name(c, f"U+{ord(c):04X}")
                for c in cluster
                if unicodedata.category(c).startswith("M")
            ]

            units.append(
                PronunciationUnit(
                    index=unit_index,
                    word_index=word_index,
                    text=cluster,
                    script=cluster_script,
                    codepoints=[f"U+{ord(c):04X}" for c in cluster],
                    unicode_names=names,
                    modifiers=modifiers,
                    has_lum_iyek=LUM_IYEK in cluster,
                    has_apun_iyek=APUN_IYEK in cluster,
                    has_virama=MEETEI_VIRAMA in cluster,
                    alignment_key=_alignment_key(cluster, cluster_script),
                )
            )
            unit_indexes.append(unit_index)

        words.append(
            PronunciationWord(
                index=word_index,
                text=word_text,
                start_char=start_char,
                end_char=end_char,
                unit_indexes=unit_indexes,
            )
        )

    warnings: list[str] = []
    if not normalized:
        warnings.append("No text was provided.")
    elif script == "mixed":
        warnings.append(
            "Mixed Meitei Mayek and Latin text detected. V1 preserves both scripts "
            "but does not guess cross-script pronunciation."
        )
    elif script == "unknown":
        warnings.append(
            "No Meitei Mayek or Latin letters were detected. Pronunciation units may be incomplete."
        )

    if script in {"latin", "mixed"}:
        warnings.append(
            "Romanized Manipuri spelling can be pronunciation-ambiguous; the native reading "
            "will remain the pronunciation authority."
        )

    representation_status = "orthographic_alignment_units"
    return {
        "language": "mni",
        "script": script,
        "normalized_text": normalized,
        "words": [asdict(word) for word in words],
        "units": [asdict(unit) for unit in units],
        "unit_sequence": [unit.alignment_key for unit in units],
        "representation": {
            "status": representation_status,
            "scheme": "voicebox-manipuri-orthographic-v1",
            "phoneme_mapping": "pending_validated_g2p",
            "native_reading_is_pronunciation_authority": True,
        },
        "markers": {
            "lum_iyek_count": sum(unit.has_lum_iyek for unit in units),
            "apun_iyek_count": sum(unit.has_apun_iyek for unit in units),
            "virama_count": sum(unit.has_virama for unit in units),
        },
        "warnings": warnings,
    }


def detect_reference_speech_bounds(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = 20,
    relative_threshold_db: float = -30.0,
    absolute_threshold_db: float = -50.0,
    padding_ms: int = 80,
) -> dict:
    """Estimate the active speech envelope without pretending to word-align.

    This is intentionally model-free.  V1 uses it to remove leading/trailing
    dead air and to validate that a native reading contains usable speech.
    """

    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = len(signal) / sample_rate if sample_rate > 0 else 0.0

    if sample_rate <= 0 or len(signal) == 0:
        return {
            "status": "no_audio",
            "start_sec": 0.0,
            "end_sec": 0.0,
            "duration_sec": duration,
            "active_duration_sec": 0.0,
            "peak": 0.0,
            "rms": 0.0,
        }

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = int(np.ceil(len(signal) / frame_len))
    rms_values = np.empty(frame_count, dtype=np.float32)

    for i in range(frame_count):
        frame = signal[i * frame_len : min(len(signal), (i + 1) * frame_len)]
        rms_values[i] = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0

    peak_rms = float(rms_values.max(initial=0.0))
    absolute_threshold = 10 ** (absolute_threshold_db / 20)
    relative_threshold = peak_rms * (10 ** (relative_threshold_db / 20))
    threshold = max(absolute_threshold, relative_threshold)

    active = np.flatnonzero(rms_values >= threshold)
    peak = float(np.max(np.abs(signal), initial=0.0))
    overall_rms = float(np.sqrt(np.mean(signal * signal)))

    if len(active) == 0:
        return {
            "status": "no_speech_detected",
            "start_sec": 0.0,
            "end_sec": duration,
            "duration_sec": duration,
            "active_duration_sec": 0.0,
            "peak": peak,
            "rms": overall_rms,
        }

    padding_frames = int(np.ceil(padding_ms / frame_ms))
    first_frame = max(0, int(active[0]) - padding_frames)
    last_frame = min(frame_count - 1, int(active[-1]) + padding_frames)

    start_sample = first_frame * frame_len
    end_sample = min(len(signal), (last_frame + 1) * frame_len)

    start_sec = start_sample / sample_rate
    end_sec = end_sample / sample_rate

    return {
        "status": "speech_detected",
        "start_sec": round(start_sec, 4),
        "end_sec": round(end_sec, 4),
        "duration_sec": round(duration, 4),
        "active_duration_sec": round(max(0.0, end_sec - start_sec), 4),
        "peak": peak,
        "rms": overall_rms,
    }
