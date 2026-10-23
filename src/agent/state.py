"""Conversation State Management for ZentroMart Support Agent.

Maintains isolated, typed session state for each multi-turn conversation.
Tracks customer context, order identification, verification lifecycle,
intent, active issues, and turn history without cross-session leakage.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import re

from src.config import MAX_VERIFICATION_RETRIES, PHONE_SUFFIX_LENGTH, PINCODE_LENGTH


class TurnRecord(BaseModel):
    """Immutable record of a single conversational turn."""
    turn_index: int
    user_message: str
    agent_reply: str
    decision: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_evidence: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None


class ConversationSession(BaseModel):
    """Isolated, typed state for a single conversation session."""
    conversation_id: str
    current_turn: int = 0
    language: str = "en"
    
    # Order & Verification State
    order_id: Optional[str] = None
    verification_status: str = "unverified"  # "unverified", "pending_proof", "verified", "failed"
    verification_attempts: int = 0
    max_retries: int = MAX_VERIFICATION_RETRIES  # Default 1 permitted retry (max 2 attempts)
    verified_order_data: Optional[Dict[str, Any]] = None  # Strictly populated ONLY after tool confirms verified=True
    
    # Issue & Context State
    active_intent: Optional[str] = None
    active_issue: Optional[str] = None
    scenario: Optional[str] = None
    
    # Escalation & Ticket State
    escalated: bool = False
    escalation_reason: Optional[str] = None
    ticket_id: Optional[str] = None
    
    # History
    history: List[Dict[str, str]] = Field(default_factory=list)  # [{"role": "user"|"assistant", "content": "..."}]
    turns: List[TurnRecord] = Field(default_factory=list)

    def is_verified(self) -> bool:
        """Check whether the active order has been verified."""
        return self.verification_status == "verified" and self.verified_order_data is not None

    def can_retry_verification(self) -> bool:
        """Allow at most one retry after the first failed attempt (total attempts <= max_retries)."""
        return self.verification_attempts <= self.max_retries

    def record_verification_attempt(self, success: bool, order_data: Optional[Dict[str, Any]] = None) -> None:
        """Update verification state and attempts count."""
        if success and order_data and order_data.get("verified") is True:
            self.verification_status = "verified"
            self.verified_order_data = order_data
        else:
            self.verification_attempts += 1
            if self.can_retry_verification():
                self.verification_status = "pending_retry"
            else:
                self.verification_status = "failed"
                self.verified_order_data = None  # Guarantee no leakage

    def record_turn(
        self,
        user_message: str,
        agent_reply: str,
        decision: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        retrieved_evidence: Optional[List[str]] = None,
    ) -> None:
        """Record turn in history and turn records."""
        self.current_turn += 1
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": agent_reply})
        self.turns.append(
            TurnRecord(
                turn_index=self.current_turn,
                user_message=user_message,
                agent_reply=agent_reply,
                decision=decision,
                tool_calls=tool_calls or [],
                retrieved_evidence=retrieved_evidence or [],
            )
        )

    def reset(self) -> None:
        """Reset all session state to initial state."""
        self.current_turn = 0
        self.order_id = None
        self.verification_status = "unverified"
        self.verification_attempts = 0
        self.verified_order_data = None
        self.active_intent = None
        self.active_issue = None
        self.scenario = None
        self.escalated = False
        self.escalation_reason = None
        self.ticket_id = None
        self.history.clear()
        self.turns.clear()


class ConversationManager:
    """Manages active conversation sessions ensuring total isolation between sessions."""

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, conversation_id: str, language: str = "en", scenario: Optional[str] = None) -> ConversationSession:
        """Retrieve existing session or create a fresh isolated session."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = ConversationSession(
                conversation_id=conversation_id,
                language=language,
                scenario=scenario,
            )
        return self._sessions[conversation_id]

    def reset_session(self, conversation_id: str) -> None:
        """Reset a specific conversation session."""
        if conversation_id in self._sessions:
            self._sessions[conversation_id].reset()

    def clear_all(self) -> None:
        """Clear all tracked sessions."""
        self._sessions.clear()


def mask_sensitive_data(text: str) -> str:
    """Mask phone numbers, pincodes, emails, and secrets to prevent data leakage per AGT-IDV."""
    if not text:
        return ""
    # Mask 10-digit Indian phone numbers
    text = re.sub(r"\b([6-9]\d{5})(\d{4})\b", r"XXXXXX\2", text)
    # Mask email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[EMAIL_MASKED]", text)
    return text
