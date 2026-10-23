"""Tests for Stage 4: Gnani.ai Speech Integration (STT & TTS) and Voice Pipeline.

Tests:
1. Audio format validation (PCM WAV, mono, <= 60 seconds).
2. Audio format error handling (non-WAV, corrupt header, duration > 60s).
3. VoiceFormatter speech rules [STY-3]:
   - 2-3 short sentences.
   - Dates with month names ("the 24th of October").
   - 8-digit order IDs in 4-digit groups ("8978 2837").
   - Currency in rupees ("forty-nine rupees", "ten thousand rupees").
   - Markdown stripping.
4. GnaniClient model validation (strictly timbre-v2.5 or timbre-v3.0).
5. GnaniClient missing credentials error.
6. Mock VoicePipeline execution and structured VoiceTurnResult.
7. Live Gnani STT & TTS integration roundtrip (when credentials present).
8. Batch audio runner execution and manifest generation.
"""
import os
import io
import wave
import pytest
from pathlib import Path

from src.config import (
    DEFAULT_STT_LANGUAGE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    TTS_MODEL_V25,
    TTS_MODEL_V30,
)
from src.voice.client import GnaniClient, validate_wav_audio
from src.voice.formatter import VoiceFormatter
from src.voice.pipeline import VoicePipeline, VoiceTurnResult
from run import run_batch_audio_pipeline


# -------------------------------------------------------------
# 1 & 2. Audio Validation Tests
# -------------------------------------------------------------
def test_validate_wav_audio_valid():
    # Generate 1-second 16kHz 16-bit mono WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000)
    wav_bytes = buf.getvalue()
    
    meta = validate_wav_audio(wav_bytes)
    assert meta["valid"] is True
    assert meta["num_channels"] == 1
    assert meta["sample_width_bytes"] == 2
    assert meta["frame_rate_hz"] == 16000
    assert meta["duration_seconds"] == 1.0


def test_validate_wav_audio_non_wav_fails():
    with pytest.raises(ValueError, match="RIFF WAV"):
        validate_wav_audio(b"NOT_A_WAV_HEADER_DATA")


def test_validate_wav_audio_duration_exceeded_fails():
    # 65 seconds audio (exceeds 60s limit)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * (16000 * 65))
    wav_bytes = buf.getvalue()
    
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        validate_wav_audio(wav_bytes, max_duration_seconds=60.0)


# -------------------------------------------------------------
# 3. VoiceFormatter Tests ([STY-3])
# -------------------------------------------------------------
def test_voice_formatter_order_id_grouping():
    raw = "Your order 89782837 is on the way."
    formatted = VoiceFormatter.format_for_speech(raw)
    assert "8978 2837" in formatted
    assert "89782837" not in formatted


def test_voice_formatter_currency_amounts():
    raw = "A reverse pickup fee of ₹49 applies. Total is ₹10,000."
    formatted = VoiceFormatter.format_for_speech(raw)
    assert "forty-nine rupees" in formatted or "49 rupees" in formatted
    assert "ten thousand rupees" in formatted
    assert "₹" not in formatted


def test_voice_formatter_dates_with_month():
    raw1 = "Your order will arrive on 2026-10-24."
    formatted1 = VoiceFormatter.format_for_speech(raw1)
    assert "the 24th of October 2026" in formatted1
    
    raw2 = "Return window closes on 13 November 2026."
    formatted2 = VoiceFormatter.format_for_speech(raw2)
    assert "the 13th of November 2026" in formatted2


def test_voice_formatter_markdown_stripping_and_sentences():
    raw = "**Hello!** This is *sentence one*. Here is sentence two. Here is sentence three. Here is sentence four."
    formatted = VoiceFormatter.format_for_speech(raw)
    assert "**" not in formatted
    assert "*" not in formatted
    # Check max 3 sentences
    assert "sentence four" not in formatted


# -------------------------------------------------------------
# 4 & 5. GnaniClient Model & Credential Validation
# -------------------------------------------------------------
def test_gnani_client_model_versioning_enforced():
    client = GnaniClient(mock_mode=True)
    # Valid models
    client.synthesize_speech("Hello", model=TTS_MODEL_V25)
    client.synthesize_speech("Hello", model=TTS_MODEL_V30)
    
    # Invalid model must raise ValueError per capstone requirement
    with pytest.raises(ValueError, match="Invalid model"):
        client.synthesize_speech("Hello", model="unsupported-model-v1")


def test_gnani_client_missing_credentials_raises():
    with pytest.raises(ValueError, match="Missing Gnani.ai API credential"):
        GnaniClient(api_key="", mock_mode=False)


# -------------------------------------------------------------
# 6. Mock Voice Pipeline End-to-End Execution
# -------------------------------------------------------------
def test_mock_voice_pipeline_execution(tmp_path):
    mock_client = GnaniClient(mock_mode=True)
    pipeline = VoicePipeline(gnani_client=mock_client)
    
    synthetic_wav = GnaniClient._generate_synthetic_wav(duration_seconds=1.0)
    out_audio = tmp_path / "reply.wav"
    
    result, wav_out = pipeline.process_turn(
        audio_input=synthetic_wav,
        session_id="test_voice_session",
        output_audio_path=out_audio,
    )
    
    assert isinstance(result, VoiceTurnResult)
    assert result.customer_transcript != ""
    assert result.spoken_text != ""
    assert out_audio.exists()
    assert len(wav_out) > 0
    assert wav_out.startswith(b"RIFF")


# -------------------------------------------------------------
# 7. Live Gnani STT & TTS Integration Test
# -------------------------------------------------------------
def test_live_gnani_speech_roundtrip():
    api_key = os.getenv("GNANI_API_KEY")
    if not api_key:
        pytest.skip("GNANI_API_KEY not configured in environment.")
        
    client = GnaniClient(api_key=api_key)
    
    # 1. Live TTS synthesis
    speech_text = "Hello, I am ZentroMart's AI support assistant."
    wav_bytes = client.synthesize_speech(
        text=speech_text,
        voice=DEFAULT_TTS_VOICE,
        model=TTS_MODEL_V25,
        language="en-IN",
    )
    assert len(wav_bytes) > 0
    assert wav_bytes.startswith(b"RIFF")
    
    # 2. Live STT transcription of the synthesized audio
    stt_res = client.transcribe_audio(wav_bytes, language_code="en-IN")
    assert stt_res["success"] is True
    assert len(stt_res["transcript"]) > 0
    # Transcript should contain core keywords
    assert "zentromart" in stt_res["transcript"].lower()


# -------------------------------------------------------------
# 8. Batch CLI Runner Test
# -------------------------------------------------------------
def test_batch_audio_pipeline_execution(tmp_path):
    audio_dir = Path("data/audio_samples")
    if not audio_dir.exists() or not list(audio_dir.glob("*.wav")):
        pytest.skip("data/audio_samples does not exist or has no audio files.")
        
    out_dir = tmp_path / "batch_out"
    results = run_batch_audio_pipeline(
        input_dir=audio_dir,
        output_dir=out_dir,
        session_id="test_batch_cli",
    )
    
    assert len(results) == 4
    assert (out_dir / "batch_manifest.csv").exists()
    assert (out_dir / "turn1_greeting_reply.wav").exists()
    assert (out_dir / "turn1_greeting_result.json").exists()
