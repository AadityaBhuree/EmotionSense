"""Multi-Speaker Diarization and Dyadic Interaction Router."""

import base64
from fastapi import APIRouter, HTTPException

from src.core.types import DiarizationResult, SpeakerTurn
from src.fusion.interaction_dynamics import DyadicInteractionAnalyzer
from src.audio.diarizer import AcousticDiarizer
from src.api.v1.schemas import DyadicAnalysisRequest, AudioDiarizeRequest

router = APIRouter(tags=["Dyadic Interaction"])


@router.post("/api/analyze/dyadic")
async def analyze_dyadic_interaction(request: DyadicAnalysisRequest):
    """Computes interpersonal synchrony, conversational balance, and Dyadic Rapport Score."""
    try:
        diar_res = None
        if request.diarization:
            turns = [
                SpeakerTurn(
                    speaker_id=t.get("speaker_id", "Speaker_0"),
                    start_time=float(t.get("start_time", 0.0)),
                    end_time=float(t.get("end_time", 0.0)),
                    duration=float(t.get("duration", 0.0)),
                )
                for t in request.diarization.get("turns", [])
            ]
            diar_res = DiarizationResult(
                turns=turns,
                speakers=request.diarization.get("speakers", []),
                speaker_durations=request.diarization.get("speaker_durations", {}),
                dominance_ratios=request.diarization.get("dominance_ratios", {}),
                interruption_count=int(request.diarization.get("interruption_count", 0)),
                total_speech_duration=float(request.diarization.get("total_speech_duration", 0.0)),
                total_audio_duration=float(request.diarization.get("total_audio_duration", 0.0)),
            )
        analyzer = DyadicInteractionAnalyzer()
        metrics = analyzer.analyze(request.participant_a_samples, request.participant_b_samples, diar_res)
        return {
            "status": "success",
            "metrics": metrics.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dyadic analysis failed: {str(exc)}")


@router.post("/api/audio/diarize")
async def diarize_audio(request: AudioDiarizeRequest):
    """Performs acoustic speaker diarization on base64 encoded audio."""
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        diarizer = AcousticDiarizer(sample_rate=request.sample_rate or 16000, num_speakers=request.num_speakers or 2)
        res = diarizer.diarize_file(audio_bytes)
        return {
            "status": "success",
            "diarization": res.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speaker diarization failed: {str(exc)}")
