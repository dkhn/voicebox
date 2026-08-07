import numpy as np
import pytest

from backend.services.ctc_forced_alignment import (
    CTCAlignmentError,
    force_align_ctc,
    minimum_ctc_frames,
    spans_to_seconds,
)


def _log_probs(best_ids: list[int], vocab_size: int) -> np.ndarray:
    probs = np.full((len(best_ids), vocab_size), 0.05 / (vocab_size - 1), dtype=np.float64)
    for frame, token_id in enumerate(best_ids):
        probs[frame, :] = 0.05 / (vocab_size - 1)
        probs[frame, token_id] = 0.95
    return np.log(probs)


def test_force_align_ctc_maps_exact_target_tokens_to_frames():
    log_probs = _log_probs([0, 1, 0, 2, 0], vocab_size=3)

    spans = force_align_ctc(log_probs, [1, 2], blank_id=0)

    assert [(span.token_id, span.start_frame, span.end_frame) for span in spans] == [
        (1, 1, 2),
        (2, 3, 4),
    ]
    assert all(span.confidence == pytest.approx(0.95) for span in spans)


def test_repeated_ctc_labels_require_an_intervening_blank():
    assert minimum_ctc_frames([1, 1]) == 3

    log_probs = _log_probs([1, 0, 1], vocab_size=2)
    spans = force_align_ctc(log_probs, [1, 1], blank_id=0)

    assert [(span.start_frame, span.end_frame) for span in spans] == [(0, 1), (2, 3)]


def test_alignment_rejects_too_few_frames_for_repeated_labels():
    log_probs = _log_probs([1, 1], vocab_size=2)

    with pytest.raises(CTCAlignmentError, match="at least 3 acoustic frames"):
        force_align_ctc(log_probs, [1, 1], blank_id=0)


def test_alignment_rejects_blank_inside_target():
    log_probs = _log_probs([0, 1, 0], vocab_size=2)

    with pytest.raises(CTCAlignmentError, match="must not contain the CTC blank"):
        force_align_ctc(log_probs, [1, 0], blank_id=0)


def test_spans_convert_to_reference_audio_seconds():
    log_probs = _log_probs([0, 1, 1, 0, 2, 0], vocab_size=3)
    spans = force_align_ctc(log_probs, [1, 2], blank_id=0)

    timed = spans_to_seconds(spans, frame_hop_sec=0.02, time_offset_sec=0.5)

    assert timed[0]["start_sec"] == pytest.approx(0.52)
    assert timed[0]["end_sec"] == pytest.approx(0.56)
    assert timed[1]["start_sec"] == pytest.approx(0.58)
    assert timed[1]["end_sec"] == pytest.approx(0.6)
