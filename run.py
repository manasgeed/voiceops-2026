"""Reproducible Batch Audio Processing CLI for ZentroMart VoiceOps 2026.

Processes a folder of customer audio files through the complete end-to-end Voice Pipeline:
1. Validates input audio (WAV, 16-bit mono PCM, <= 60 seconds).
2. Speech-to-Text transcription via Gnani Prisma v2.5.
3. Conversational reasoning, identity verification, tools, and RAG grounding.
4. Voice formatting per AGT-STY [STY-3].
5. Text-to-Speech synthesis via Gnani Timbre v2.5 / v3.0.
6. Saves generated speech audio, transcribed text, agent reply, and retrieved evidence.

Usage:
    python run.py --input-dir <input_folder> --output-dir <output_folder>
"""
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_STT_LANGUAGE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    RESULTS_DIR,
)
from src.voice.client import GnaniClient
from src.voice.pipeline import VoicePipeline, VoiceTurnResult
from src.agent.conversational_agent import ConversationalAgent


def run_batch_audio_pipeline(
    input_dir: Path,
    output_dir: Path,
    session_id: str = "batch_session",
    language_code: str = DEFAULT_STT_LANGUAGE,
    voice: str = DEFAULT_TTS_VOICE,
    tts_model: str = DEFAULT_TTS_MODEL,
) -> list[VoiceTurnResult]:
    """Process all WAV files in input_dir through the voice pipeline."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
        
    wav_files = sorted(list(input_path.glob("*.wav")))
    if not wav_files:
        print(f"No .wav files found in {input_path}")
        return []
        
    print(f"\n========================================================")
    print(f"VOICEOPS 2026: BATCH AUDIO PIPELINE")
    print(f"========================================================")
    print(f"Input Directory:     {input_path}")
    print(f"Output Directory:    {output_path}")
    print(f"Total Audio Files:   {len(wav_files)}")
    print(f"Language:            {language_code}")
    print(f"TTS Model:           {tts_model}")
    print(f"TTS Voice:           {voice}")
    print(f"Session ID:          {session_id}")
    print(f"========================================================\n")

    client = GnaniClient()
    agent = ConversationalAgent()
    pipeline = VoicePipeline(gnani_client=client, agent=agent)
    
    # Reset conversation session for batch run
    agent.manager.reset_session(session_id)
    
    results: list[VoiceTurnResult] = []
    manifest_records: list[dict] = []
    
    for idx, audio_file in enumerate(wav_files, 1):
        stem = audio_file.stem
        out_audio_file = output_path / f"{stem}_reply.wav"
        out_json_file = output_path / f"{stem}_result.json"
        
        print(f"[{idx}/{len(wav_files)}] Processing {audio_file.name}...")
        try:
            turn_result, _ = pipeline.process_turn(
                audio_input=audio_file,
                session_id=session_id,
                language_code=language_code,
                voice=voice,
                tts_model=tts_model,
                output_audio_path=out_audio_file,
            )
            
            # Save per-turn structured result JSON
            with open(out_json_file, "w", encoding="utf-8") as f:
                json.dump(turn_result.model_dump(), f, indent=2)
                
            results.append(turn_result)
            
            manifest_records.append({
                "audio_input": audio_file.name,
                "customer_transcript": turn_result.customer_transcript,
                "decision": turn_result.decision,
                "agent_reply": turn_result.agent_reply_text,
                "spoken_text": turn_result.spoken_text,
                "tool_calls": " ; ".join(tc.get("call_str", tc.get("tool", "")) for tc in turn_result.tool_calls),
                "retrieved_evidence": " ; ".join(turn_result.retrieved_evidence),
                "output_audio": out_audio_file.name,
                "latency_sec": turn_result.total_latency_seconds,
            })
            
            print(f"  Transcript: \"{turn_result.customer_transcript}\"")
            print(f"  Decision:   {turn_result.decision}")
            print(f"  Reply:      \"{turn_result.agent_reply_text[:80]}...\"")
            print(f"  Evidence:   {turn_result.retrieved_evidence}")
            print(f"  Audio Out:  {out_audio_file.name} (Total: {turn_result.total_latency_seconds}s)")
            print()
        except Exception as e:
            print(f"  [ERROR] Failed to process {audio_file.name}: {e}\n")
            
    # Save batch summary manifest CSV
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = output_path / "batch_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False, encoding="utf-8")
    print(f"Batch execution complete. Summary saved to: {manifest_csv}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VoiceOps 2026 Batch Audio Processor.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=str, default="data/audio_samples", help="Input directory of customer WAV files.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="results/voice_output", help="Output directory for generated speech & results.")
    parser.add_argument("--session-id", "--session_id", dest="session_id", type=str, default="batch_session", help="Session ID for state tracking.")
    parser.add_argument("--language", type=str, default=DEFAULT_STT_LANGUAGE, help="Language code (e.g. en-IN, hi-IN).")
    parser.add_argument("--voice", type=str, default=DEFAULT_TTS_VOICE, help="TTS voice name (e.g. Kaveri).")
    parser.add_argument("--model", type=str, default=DEFAULT_TTS_MODEL, help="TTS model (timbre-v2.5 or timbre-v3.0).")
    args = parser.parse_args()

    run_batch_audio_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        session_id=args.session_id,
        language_code=args.language,
        voice=args.voice,
        tts_model=args.model,
    )


if __name__ == "__main__":
    main()
