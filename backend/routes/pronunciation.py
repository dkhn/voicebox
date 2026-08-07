"""Pronunciation-guide endpoints for low-resource language re-singing."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..services.manipuri_pronunciation import (
    build_manipuri_pronunciation_plan,
    detect_reference_speech_bounds,
)
from ..utils.audio import load_audio


router = APIRouter(prefix="/pronunciation", tags=["pronunciation"])

UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_REFERENCE_BYTES = 100 * 1024 * 1024
MAX_REFERENCE_DURATION_SEC = 15 * 60
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}


class ManipuriAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


@router.post("/manipuri/analyze")
async def analyze_manipuri_text(request: ManipuriAnalyzeRequest):
    """Build deterministic Manipuri pronunciation/alignment units from lyrics."""

    return build_manipuri_pronunciation_plan(request.text)


@router.post("/manipuri/reference")
async def prepare_manipuri_native_reference(
    text: str = Form(..., min_length=1, max_length=50_000),
    file: UploadFile = File(...),
):
    """Prepare a native Manipuri reading for later word/phoneme alignment.

    V1 validates and analyzes the guide recording, strips the problem down to
    a stable text-unit sequence, and reports the active speech envelope.  It
    intentionally does not fabricate word timestamps before a validated
    acoustic aligner is connected.
    """

    uploaded_ext = Path(file.filename or "").suffix.lower()
    if uploaded_ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{uploaded_ext or 'unknown'}'. "
            f"Supported: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}",
        )

    total_bytes = 0
    digest = hashlib.sha256()

    with tempfile.NamedTemporaryFile(suffix=uploaded_ext, delete=False) as tmp:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > MAX_REFERENCE_BYTES:
                tmp_path = tmp.name
                Path(tmp_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="Native pronunciation reference is larger than 100 MB.",
                )
            digest.update(chunk)
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        audio, sample_rate = await asyncio.to_thread(
            load_audio,
            tmp_path,
            16_000,
            True,
        )
        duration_sec = len(audio) / sample_rate

        if duration_sec <= 0:
            raise HTTPException(status_code=400, detail="The uploaded audio contains no samples.")
        if duration_sec > MAX_REFERENCE_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail="Native pronunciation reference must be 15 minutes or shorter.",
            )

        speech_bounds = detect_reference_speech_bounds(audio, sample_rate)
        if speech_bounds["status"] != "speech_detected":
            raise HTTPException(
                status_code=400,
                detail="No usable speech was detected in the native pronunciation reference.",
            )

        plan = build_manipuri_pronunciation_plan(text)

        return {
            **plan,
            "reference_audio": {
                "filename": file.filename,
                "sha256": digest.hexdigest(),
                "size_bytes": total_bytes,
                "sample_rate": sample_rate,
                "duration_sec": round(duration_sec, 4),
                "speech_bounds": speech_bounds,
            },
            "alignment": {
                "status": "prepared",
                "word_timestamps": None,
                "unit_timestamps": None,
                "engine": None,
                "next_step": (
                    "Attach a validated acoustic forced aligner to map the provided "
                    "native reading onto words and pronunciation units."
                ),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze reference audio: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
