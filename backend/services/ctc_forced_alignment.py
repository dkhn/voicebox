"""Small, model-agnostic CTC forced-alignment core.

The acoustic model is deliberately not coupled to this module.  Any future
Manipuri CTC encoder can provide frame-level log probabilities plus token ids;
this dynamic-programming aligner then constrains the acoustic path to the
*exact lyrics supplied by the user*.

That separation matters for low-resource language work: ASR transcription may
rewrite or normalize words, while forced alignment must preserve the original
Meitei/Manipuri text as the linguistic source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CTCTokenSpan:
    """Aligned frame span for one target token."""

    target_index: int
    token_id: int
    start_frame: int
    end_frame: int
    confidence: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict:
        return {**asdict(self), "frame_count": self.frame_count}


class CTCAlignmentError(ValueError):
    """Raised when an emission cannot contain the requested CTC target."""


def minimum_ctc_frames(target_ids: Sequence[int]) -> int:
    """Minimum frames required for a CTC path over *target_ids*.

    Consecutive repeated labels require an intervening blank frame.
    """

    if not target_ids:
        return 0
    repeats = sum(a == b for a, b in zip(target_ids, target_ids[1:]))
    return len(target_ids) + repeats


def _validate_inputs(log_probs: np.ndarray, target_ids: Sequence[int], blank_id: int) -> None:
    if log_probs.ndim != 2:
        raise CTCAlignmentError(
            f"log_probs must have shape [frames, vocabulary], got {tuple(log_probs.shape)}"
        )
    if log_probs.shape[0] == 0:
        raise CTCAlignmentError("log_probs contains no acoustic frames")
    if log_probs.shape[1] == 0:
        raise CTCAlignmentError("log_probs contains an empty vocabulary")
    if not 0 <= blank_id < log_probs.shape[1]:
        raise CTCAlignmentError(f"blank_id {blank_id} is outside the vocabulary")
    if any(token_id == blank_id for token_id in target_ids):
        raise CTCAlignmentError("target_ids must not contain the CTC blank token")
    if any(token_id < 0 or token_id >= log_probs.shape[1] for token_id in target_ids):
        raise CTCAlignmentError("target_ids contains a token outside the vocabulary")

    required = minimum_ctc_frames(target_ids)
    if log_probs.shape[0] < required:
        raise CTCAlignmentError(
            f"Need at least {required} acoustic frames for this CTC target, "
            f"but only {log_probs.shape[0]} were provided"
        )


def _extended_target(target_ids: Sequence[int], blank_id: int) -> np.ndarray:
    states = np.full((2 * len(target_ids) + 1,), blank_id, dtype=np.int64)
    if target_ids:
        states[1::2] = np.asarray(target_ids, dtype=np.int64)
    return states


def force_align_ctc(
    log_probs: np.ndarray,
    target_ids: Sequence[int],
    *,
    blank_id: int,
) -> list[CTCTokenSpan]:
    """Force-align exact CTC targets against frame-level log probabilities.

    Parameters
    ----------
    log_probs:
        ``[T, V]`` log probabilities (natural log) from a CTC acoustic model.
    target_ids:
        Exact non-blank token sequence to preserve.
    blank_id:
        Vocabulary id used by the acoustic model for CTC blank.

    Returns
    -------
    list[CTCTokenSpan]
        One monotonically ordered span per target token. ``end_frame`` is
        exclusive. Confidence is the geometric mean frame probability for the
        token while the Viterbi path occupies its non-blank state.
    """

    emissions = np.asarray(log_probs, dtype=np.float64)
    target = [int(token_id) for token_id in target_ids]
    _validate_inputs(emissions, target, blank_id)

    if not target:
        return []

    states = _extended_target(target, blank_id)
    frame_count = emissions.shape[0]
    state_count = len(states)
    neg_inf = -np.inf

    scores = np.full((frame_count, state_count), neg_inf, dtype=np.float64)
    backpointers = np.full((frame_count, state_count), -1, dtype=np.int32)

    # At t=0 a valid CTC path can be either the leading blank or first label.
    scores[0, 0] = emissions[0, blank_id]
    if state_count > 1:
        scores[0, 1] = emissions[0, states[1]]

    for t in range(1, frame_count):
        # Reachability grows by at most two CTC states per acoustic frame.
        max_state = min(state_count - 1, 2 * t + 1)
        min_state = max(0, state_count - 2 * (frame_count - t) - 1)

        for s in range(min_state, max_state + 1):
            current_label = int(states[s])
            candidates: list[tuple[float, int]] = [(scores[t - 1, s], s)]

            if s > 0:
                candidates.append((scores[t - 1, s - 1], s - 1))

            # Skip over a blank only when entering a non-blank state whose
            # label differs from the previous target label. Repeated labels
            # therefore require an explicit blank between them.
            if (
                s > 1
                and current_label != blank_id
                and current_label != int(states[s - 2])
            ):
                candidates.append((scores[t - 1, s - 2], s - 2))

            previous_score, previous_state = max(candidates, key=lambda item: item[0])
            if math.isinf(previous_score) and previous_score < 0:
                continue

            scores[t, s] = previous_score + emissions[t, current_label]
            backpointers[t, s] = previous_state

    # A complete CTC path may finish on the last label or trailing blank.
    final_candidates = [state_count - 1]
    if state_count > 1:
        final_candidates.append(state_count - 2)
    final_state = max(final_candidates, key=lambda s: scores[-1, s])

    if np.isneginf(scores[-1, final_state]):
        raise CTCAlignmentError("No valid CTC path could align the requested target")

    state_path = np.empty(frame_count, dtype=np.int32)
    state_path[-1] = final_state
    for t in range(frame_count - 1, 0, -1):
        previous = backpointers[t, state_path[t]]
        if previous < 0:
            raise CTCAlignmentError("CTC backtrace reached an unreachable state")
        state_path[t - 1] = previous

    spans: list[CTCTokenSpan] = []
    for target_index, token_id in enumerate(target):
        state_index = 2 * target_index + 1
        frames = np.flatnonzero(state_path == state_index)
        if len(frames) == 0:
            raise CTCAlignmentError(
                f"Aligned path did not visit target token index {target_index}"
            )

        start_frame = int(frames[0])
        end_frame = int(frames[-1]) + 1
        token_frame_scores = emissions[frames, token_id]
        confidence = float(np.exp(np.mean(token_frame_scores)))

        spans.append(
            CTCTokenSpan(
                target_index=target_index,
                token_id=token_id,
                start_frame=start_frame,
                end_frame=end_frame,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )

    return spans


def spans_to_seconds(
    spans: Sequence[CTCTokenSpan],
    *,
    frame_hop_sec: float,
    time_offset_sec: float = 0.0,
) -> list[dict]:
    """Convert CTC frame spans into time ranges for API responses."""

    if frame_hop_sec <= 0:
        raise ValueError("frame_hop_sec must be greater than zero")

    output: list[dict] = []
    for span in spans:
        item = span.to_dict()
        item.update(
            {
                "start_sec": round(time_offset_sec + span.start_frame * frame_hop_sec, 6),
                "end_sec": round(time_offset_sec + span.end_frame * frame_hop_sec, 6),
            }
        )
        output.append(item)
    return output
