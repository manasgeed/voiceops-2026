"""Stage 2 Document-Grounded RAG Agent for ZentroMart Customer Support.

Retrieves authoritative policy and catalog documents, constructs a grounded context,
and generates concise, traceable customer-support responses citing genuine section IDs.
Strictly does NOT invoke order lookup or ticket creation tools (reserved for Stage 3).
"""
import os
import time
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from src.config import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, ENV_FILE
from src.baseline.agent import AgentResponse, parse_decision_and_reply

# Ensure root .env is loaded
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.rag.prompts import RAG_SYSTEM_PROMPT, format_rag_user_prompt

INSUFFICIENT_EVIDENCE_REPLY = (
    "Hello! I am ZentroMart's AI support assistant. "
    "Sorry, I don't have information about that in our policy documents. "
    "I can connect you to a colleague who can help."
)

class RAGAgent:
    """Stage 2 Document-Grounded Support Agent."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        client: Optional[object] = None,
        retriever: Optional[HybridRetriever] = None,
        top_k: int = 5,
        min_relevance_threshold: float = 0.012,
    ):
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_LLM_MODEL
        self.temperature = temperature
        self.top_k = top_k
        self.min_relevance_threshold = min_relevance_threshold
        
        self.retriever = retriever if retriever is not None else HybridRetriever()
        
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
        """Execute Stage 2 Grounded RAG flow: Retrieve -> Check -> Generate -> Cite."""
        start_time = time.time()
        
        # 1. Retrieve relevant authoritative chunks
        retrieved_results: List[RetrievalResult] = self.retriever.retrieve(message, top_k=self.top_k)
        retrieved_section_ids = [r.section_id for r in retrieved_results]
        
        # 2. Check if evidence is sufficient
        max_score = max([r.score for r in retrieved_results]) if retrieved_results else 0.0
        if not retrieved_results or max_score < self.min_relevance_threshold:
            latency = time.time() - start_time
            return AgentResponse(
                reply_text=INSUFFICIENT_EVIDENCE_REPLY,
                decision="escalate",
                model=self.model,
                raw_output=INSUFFICIENT_EVIDENCE_REPLY,
                latency_seconds=round(latency, 3),
                retrieved_evidence=retrieved_section_ids,
                tool_calls=[],
            )

        # 3. Generate response grounded in retrieved evidence
        user_prompt = format_rag_user_prompt(message, retrieved_results)
        
        max_retries = 3
        backoff = 4.0
        for attempt in range(max_retries):
            try:
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_PROMPT,
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
                    retrieved_evidence=retrieved_section_ids,
                    tool_calls=[],
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
                    retrieved_evidence=retrieved_section_ids,
                    tool_calls=[],
                )
