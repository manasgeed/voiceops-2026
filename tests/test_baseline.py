"""Unit tests for Stage 1 Baseline Agent using mocks (Milestone 1)."""
import pytest
from unittest.mock import MagicMock
from src.baseline.agent import BaselineAgent, AgentResponse, parse_decision_and_reply
from src.baseline.prompts import BASELINE_SYSTEM_PROMPT, format_user_prompt

def test_baseline_missing_credentials_raises_value_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing Gemini API credential"):
        BaselineAgent(api_key=None, client=None)

def test_parse_decision_and_reply_explicit_decision():
    raw_text = (
        "Hello! I am ZentroMart's AI support assistant. How can I help you?\n"
        "DECISION: answer"
    )
    reply, decision = parse_decision_and_reply(raw_text)
    assert "ZentroMart's AI support assistant" in reply
    assert decision == "answer"
    assert "DECISION:" not in reply

def test_parse_decision_and_reply_fallback_heuristics():
    # Fallback to escalate
    raw_escalate = "I will connect you with a human agent to review your damaged TV."
    _, decision = parse_decision_and_reply(raw_escalate)
    assert decision == "escalate"

    # Fallback to clarify
    raw_clarify = "Could you provide your order number so I can check?"
    _, decision = parse_decision_and_reply(raw_clarify)
    assert decision == "clarify"

    # Fallback to refuse
    raw_refuse = "We cannot fulfill this request as it is against policy."
    _, decision = parse_decision_and_reply(raw_refuse)
    assert decision == "refuse"

def test_baseline_agent_mocked_generation():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "Hello! I am ZentroMart's AI support assistant. Electronics have a 7-day return window.\n"
        "DECISION: answer"
    )
    mock_client.models.generate_content.return_value = mock_response

    agent = BaselineAgent(api_key="mock_key", client=mock_client)
    res = agent.respond("What is the return window for headphones?")

    assert isinstance(res, AgentResponse)
    assert "ZentroMart's AI support assistant" in res.reply_text
    assert res.decision == "answer"
    assert res.error is None
    assert res.retrieved_evidence == []  # Strictly empty in Stage 1
    assert res.tool_calls == []          # Strictly empty in Stage 1

def test_baseline_agent_error_handling():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Service Unavailable")

    agent = BaselineAgent(api_key="mock_key", client=mock_client)
    res = agent.respond("Hello")

    assert res.decision == "error"
    assert "Service Unavailable" in res.error
    assert res.reply_text == ""

def test_prompts_strictly_exclude_rag_and_gold_evidence():
    # User prompt must only wrap the raw customer question
    formatted = format_user_prompt("Can I return opened shampoo?")
    assert "Can I return opened shampoo?" in formatted
    assert "gold_evidence" not in formatted
    assert "reference_answer" not in formatted
    assert "must_include" not in formatted

    # System prompt establishes AI role and uncertainty acknowledgment
    assert "ZentroMart's AI customer support assistant" in BASELINE_SYSTEM_PROMPT
    assert "internal company policy documents" in BASELINE_SYSTEM_PROMPT
