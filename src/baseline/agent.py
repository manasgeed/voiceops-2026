"""Baseline Agent using official Google GenAI SDK.

Stage 1 Prompting-Only Agent. Strictly does NOT use RAG, document retrieval,
or mock tool calls.
"""
import os
import re
import time
from typing import Optional, Protocol, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

from src.config import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, ENV_FILE
from src.baseline.prompts import BASELINE_SYSTEM_PROMPT, format_user_prompt

# Ensure root .env is loaded
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

class AgentResponse(BaseModel):
    """Standardized response schema for VoiceOps customer support agents."""
    reply_text: str = Field(description="Customer-facing response text.")
    decision: str = Field(description="Decision category: answer, clarify, escalate, refuse.")
    model: str = Field(description="Model identifier used for generation.")
    raw_output: str = Field(default="", description="Complete raw text output from the model.")
    latency_seconds: float = Field(default=0.0, description="Inference latency in seconds.")
    error: Optional[str] = Field(default=None, description="Error message if generation failed.")
    retrieved_evidence: List[str] = Field(default_factory=list, description="Section IDs used (empty in Stage 1).")
    tool_calls: List[dict] = Field(default_factory=list, description="Tool calls executed (empty in Stage 1).")

class AgentInterface(Protocol):
    """Protocol ensuring consistent agent interface across all capstone stages."""
    def respond(self, message: str) -> AgentResponse:
        ...

def parse_decision_and_reply(raw_text: str) -> tuple[str, str]:
    """Extract customer-facing reply and decision category from raw LLM output."""
    if not raw_text:
        return "", "answer"
    
    decision = None
    reply_lines = []
    
    for line in raw_text.splitlines():
        match = re.match(r"^DECISION:\s*(answer|clarify|escalate|refuse)", line.strip(), re.IGNORECASE)
        if match:
            decision = match.group(1).lower()
        else:
            reply_lines.append(line)
            
    reply_text = "\n".join(reply_lines).strip()
    
    # Fallback heuristic if explicit tag was omitted by the LLM
    if not decision:
        lower_reply = reply_text.lower()
        if any(term in lower_reply for term in ["colleague", "human agent", "support team", "handover", "ticket"]):
            decision = "escalate"
        elif any(term in lower_reply for term in ["could you provide", "please share", "order number?", "what is your"]):
            decision = "clarify"
        elif any(term in lower_reply for term in ["cannot fulfill", "not permitted", "policy does not allow", "against policy"]):
            decision = "refuse"
        else:
            decision = "answer"
            
    return reply_text, decision

class BaselineAgent:
    """Stage 1 Baseline Support Agent using direct prompting."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        client: Optional[object] = None,
    ):
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_LLM_MODEL
        self.temperature = temperature
        
        # Read API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                raise ValueError(
                    "Missing Gemini API credential. Set GEMINI_API_KEY or GOOGLE_API_KEY in "
                    "your environment or .env file, or pass client explicitly."
                )
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            
    def respond(self, message: str) -> AgentResponse:
        """Generate response for a single customer message."""
        start_time = time.time()
        user_prompt = format_user_prompt(message)
        
        max_retries = 3
        backoff = 4.0
        for attempt in range(max_retries):
            try:
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=BASELINE_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=350,
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )
                latency = time.time() - start_time
                raw_text = response.text or ""
                reply_text, decision = parse_decision_and_reply(raw_text)
                
                return AgentResponse(
                    reply_text=reply_text,
                    decision=decision,
                    model=self.model,
                    raw_output=raw_text,
                    latency_seconds=round(latency, 3),
                    retrieved_evidence=[],  # Strictly empty in Stage 1
                    tool_calls=[],          # Strictly empty in Stage 1
                )
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str) and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                latency = time.time() - start_time
                return AgentResponse(
                    reply_text="",
                    decision="error",
                    model=self.model,
                    raw_output="",
                    latency_seconds=round(latency, 3),
                    error=str(e),
                    retrieved_evidence=[],
                    tool_calls=[],
                )
