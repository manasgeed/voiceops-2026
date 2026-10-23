"""End-to-End Voice Pipeline integrating Gnani STT, ConversationalAgent, and Gnani TTS.

Processes customer speech input:
1. Validates audio format (PCM WAV, mono, <= 60s).
2. Transcribes speech to text via Gnani Prisma v2.5.
3. Generates policy-grounded, tool-enabled response via ConversationalAgent.
4. Formats response for speech per AGT-STY [STY-3].
5. Synthesizes output speech via Gnani Timbre v2.5 / v3.0.
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_STT_LANGUAGE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
)
from src.baseline.agent import AgentResponse
from src.agent.conversational_agent import ConversationalAgent
from src.voice.client import GnaniClient
from src.voice.formatter import VoiceFormatter


class VoiceTurnResult(BaseModel):
    """Complete structured output for an end-to-end voice turn."""
    session_id: str
    input_audio: str
    customer_transcript: str
    agent_reply_text: str
    spoken_text: str
    decision: str
    tool_calls: list = Field(default_factory=list)
    retrieved_evidence: list = Field(default_factory=list)
    output_audio_path: Optional[str] = None
    stt_latency_seconds: float = 0.0
    llm_latency_seconds: float = 0.0
    tts_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0
    error: Optional[str] = None


class VoicePipeline:
    """Manages audio-to-audio conversational support turn execution."""

    def __init__(
        self,
        gnani_client: Optional[GnaniClient] = None,
        agent: Optional[ConversationalAgent] = None,
        voice_formatter: Optional[VoiceFormatter] = None,
    ):
        self.gnani = gnani_client or GnaniClient()
        self.agent = agent or ConversationalAgent()
        self.formatter = voice_formatter or VoiceFormatter()

    def process_turn(
        self,
        audio_input: Union[str, Path, bytes],
        session_id: str = "voice_session",
        language_code: str = DEFAULT_STT_LANGUAGE,
        voice: str = DEFAULT_TTS_VOICE,
        tts_model: str = DEFAULT_TTS_MODEL,
        output_audio_path: Optional[Union[str, Path]] = None,
    ) -> tuple[VoiceTurnResult, bytes]:
        """Process a single voice turn: Audio In -> STT -> Agent -> TTS -> Audio Out."""
        start_total = time.time()
        input_label = str(audio_input) if isinstance(audio_input, (str, Path)) else "<bytes>"
        
        # 1. Speech-to-Text Transcription via Gnani Prisma v2.5
        stt_start = time.time()
        stt_result = self.gnani.transcribe_audio(audio_input, language_code=language_code)
        stt_latency = time.time() - stt_start
        transcript = stt_result.get("transcript", "")
        
        # 2. Conversational Agent Execution (State, Verification, Tools, RAG Grounding)
        agent_response: AgentResponse = self.agent.respond(
            message=transcript,
            conversation_id=session_id,
            language=language_code.split("-")[0],
        )
        llm_latency = agent_response.latency_seconds
        
        # 3. Speech-optimized formatting per AGT-STY [STY-3]
        spoken_text = self.formatter.format_for_speech(agent_response.reply_text)
        
        # 4. Text-to-Speech Synthesis via Gnani Timbre v2.5 / v3.0
        tts_start = time.time()
        synthesized_wav = self.gnani.synthesize_speech(
            text=spoken_text or "Sorry, I could not process that.",
            voice=voice,
            model=tts_model,
            language=language_code,
        )
        tts_latency = time.time() - tts_start
        
        # 5. Save output audio file if path provided
        saved_path = None
        if output_audio_path:
            p = Path(output_audio_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                f.write(synthesized_wav)
            saved_path = str(p)
            
        total_latency = time.time() - start_total
        
        return VoiceTurnResult(
            session_id=session_id,
            input_audio=input_label,
            customer_transcript=transcript,
            agent_reply_text=agent_response.reply_text,
            spoken_text=spoken_text,
            decision=agent_response.decision,
            tool_calls=agent_response.tool_calls,
            retrieved_evidence=agent_response.retrieved_evidence,
            output_audio_path=saved_path,
            stt_latency_seconds=round(stt_latency, 3),
            llm_latency_seconds=round(llm_latency, 3),
            tts_latency_seconds=round(tts_latency, 3),
            total_latency_seconds=round(total_latency, 3),
        ), synthesized_wav
