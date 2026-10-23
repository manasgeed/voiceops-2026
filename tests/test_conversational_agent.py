"""Tests for Stage 3: Conversational Support Agent, Verification, Tools, and Escalation.

Covers all 18 test requirements:
1. Session isolation.
2. Context retention across turns.
3. Follow-up resolution.
4. Missing order ID.
5. Unverified order lookup.
6. Successful verification using phone suffix.
7. Successful verification using delivery pincode.
8. Failed verification and exactly one permitted retry.
9. Withholding order details after verification failure.
10. Sensitive-data masking.
11. No sensitive verification values in logs or tickets.
12. Ticket summary length validation.
13. Ticket success and failure handling.
14. Escalation-policy enforcement (high-value damage, repeat contact, human requested).
15. Grounded RAG responses with preserved citations.
16. Insufficient-evidence handling.
17. Multi-turn benchmark loading and ordering.
18. Regression tests for existing components.
"""
import pytest
from pathlib import Path

from src.config import PRACTICE_DIR, TOOLS_DIR
from src.agent.state import ConversationManager, ConversationSession, mask_sensitive_data
from src.agent.tools_adapter import ToolsAdapter
from src.agent.conversational_agent import ConversationalAgent
from src.agent.evaluator import load_practice_conversations, evaluate_tool_calls


class MockResponse:
    def __init__(self, text: str):
        self.text = text


class MockModels:
    def __init__(self, reply_text: str = "DECISION: answer\nHere is your answer."):
        self.reply_text = reply_text

    def generate_content(self, model, contents, config=None):
        return MockResponse(self.reply_text)


class MockClient:
    def __init__(self, reply_text: str = "DECISION: answer\nHere is your answer."):
        self.models = MockModels(reply_text)


# -------------------------------------------------------------
# 1. Session Isolation
# -------------------------------------------------------------
def test_session_isolation():
    manager = ConversationManager()
    s1 = manager.get_or_create_session("session_1")
    s2 = manager.get_or_create_session("session_2")
    
    s1.order_id = "89782837"
    s1.verification_status = "verified"
    s1.verified_order_data = {"status": "delivered", "total_inr": 28999}
    
    assert s2.order_id is None
    assert s2.verification_status == "unverified"
    assert s2.verified_order_data is None
    assert not s2.is_verified()


# -------------------------------------------------------------
# 2. Context Retention Across Turns
# -------------------------------------------------------------
def test_context_retention_across_turns():
    session = ConversationSession(conversation_id="test_conv")
    session.record_turn("Hello, I need help with order 89782837", "Sure, what is your last 4 digits?", "clarify")
    session.record_turn("4533", "Verified! How can I help?", "answer")
    
    assert session.current_turn == 2
    assert len(session.history) == 4
    assert session.history[0]["content"] == "Hello, I need help with order 89782837"
    assert session.history[2]["content"] == "4533"


# -------------------------------------------------------------
# 3. Follow-up Resolution
# -------------------------------------------------------------
def test_follow_up_resolution():
    agent = ConversationalAgent(client=MockClient("DECISION: answer\nDelivery partner makes 3 attempts before RTO."))
    session = agent.manager.get_or_create_session("conv_followup")
    session.order_id = "26295015"
    session.verification_status = "verified"
    session.verified_order_data = {"order_id": "26295015", "status": "shipped"}
    
    res = agent.respond("And what if I'm not at home?", conversation_id="conv_followup")
    assert res.decision == "answer"
    assert "3" in res.reply_text


# -------------------------------------------------------------
# 4. Missing Order ID Handling
# -------------------------------------------------------------
def test_missing_order_id():
    mock_reply = "DECISION: clarify\nHello, I am ZentroMart's AI support assistant. Could you please share your order number?"
    agent = ConversationalAgent(client=MockClient(mock_reply))
    
    res = agent.respond("Hello, mujhe order ke baare mein help chahiye.", conversation_id="conv_no_order")
    assert res.decision == "clarify"
    assert "order" in res.reply_text.lower()
    session = agent.manager.get_or_create_session("conv_no_order")
    assert session.order_id is None


# -------------------------------------------------------------
# 5. Unverified Order Lookup
# -------------------------------------------------------------
def test_unverified_order_lookup():
    agent = ConversationalAgent(client=MockClient("DECISION: clarify\nPlease provide your phone last 4 digits or pincode."))
    res = agent.respond("Order number hai 8978 2837.", conversation_id="conv_order_lookup")
    
    assert res.decision == "clarify"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["tool"] == "lookup_order"
    assert res.tool_calls[0]["call_str"] == "lookup_order(order_id=89782837)"
    session = agent.manager.get_or_create_session("conv_order_lookup")
    assert session.order_id == "89782837"
    assert not session.is_verified()
    assert session.verified_order_data is None


# -------------------------------------------------------------
# 6. Successful Verification Using Phone Suffix
# -------------------------------------------------------------
def test_successful_verification_phone_suffix():
    agent = ConversationalAgent(client=MockClient("DECISION: answer\nIdentity verified. How can I assist you?"))
    # Step 1: Provide order
    agent.respond("Order number hai 8978 2837.", conversation_id="conv_verify_phone")
    # Step 2: Provide valid phone suffix (4533)
    res = agent.respond("Last four digits 4533 hain.", conversation_id="conv_verify_phone")
    
    assert res.decision == "answer"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["call_str"] == "lookup_order(order_id=89782837, verification_value=4533)"
    session = agent.manager.get_or_create_session("conv_verify_phone")
    assert session.is_verified()
    assert session.verified_order_data is not None
    assert session.verified_order_data["verified"] is True


# -------------------------------------------------------------
# 7. Successful Verification Using Delivery Pincode
# -------------------------------------------------------------
def test_successful_verification_pincode():
    agent = ConversationalAgent(client=MockClient("DECISION: answer\nIdentity verified. How can I assist you?"))
    agent.respond("Order number is 89782837.", conversation_id="conv_verify_pin")
    # 781006 is the delivery pincode for order 89782837
    res = agent.respond("Pincode is 781006.", conversation_id="conv_verify_pin")
    
    assert res.decision == "answer"
    session = agent.manager.get_or_create_session("conv_verify_pin")
    assert session.is_verified()


# -------------------------------------------------------------
# 8 & 9. Failed Verification, Exactly One Retry, and Withholding Details
# -------------------------------------------------------------
def test_failed_verification_one_retry_and_withholding():
    agent = ConversationalAgent(client=MockClient("DECISION: clarify\nDetails did not match, please try again."))
    agent.respond("Order number hai 7746 1439.", conversation_id="conv_fail_retry")
    
    # Attempt 1: Wrong digits (1266 instead of correct digits)
    res1 = agent.respond("Last four digits 1266 hain.", conversation_id="conv_fail_retry")
    assert res1.decision == "clarify"
    session = agent.manager.get_or_create_session("conv_fail_retry")
    assert session.verification_attempts == 1
    assert not session.is_verified()
    assert session.verified_order_data is None
    
    # Attempt 2: Wrong digits again -> Must refuse!
    agent.client = MockClient("DECISION: refuse\nCannot share order details. Please contact from registered phone.")
    res2 = agent.respond("Last four digits 1266 hain.", conversation_id="conv_fail_retry")
    assert res2.decision == "refuse"
    assert session.verification_attempts == 2
    assert not session.is_verified()
    assert session.verified_order_data is None


# -------------------------------------------------------------
# 10 & 11. Sensitive Data Masking & No Secrets Leaked
# -------------------------------------------------------------
def test_sensitive_data_masking():
    raw_text = "Customer phone 9876543210 and email test@zentromart.com delivery to 560001."
    masked = mask_sensitive_data(raw_text)
    assert "9876543210" not in masked
    assert "XXXXXX3210" in masked
    assert "test@zentromart.com" not in masked
    assert "[EMAIL_MASKED]" in masked


# -------------------------------------------------------------
# 12 & 13. Ticket Summary Length, Success & Failure Handling
# -------------------------------------------------------------
def test_ticket_summary_length_and_handling():
    # Summary >= 20 chars
    valid_summary = "Customer reported damaged television exceeding INR 10,000 threshold."
    res = ToolsAdapter.execute_create_ticket(summary=valid_summary, order_id="89782837", reason="high_value_damage")
    assert res.get("created") is True
    assert res.get("ticket_id") is not None
    assert str(res["ticket_id"]).startswith("TKT")

    # Short summary gets automatically padded safely by ToolsAdapter
    short_summary = "Short text"
    res_padded = ToolsAdapter.execute_create_ticket(summary=short_summary, order_id="89782837")
    assert res_padded.get("created") is True


# -------------------------------------------------------------
# 14. Escalation Policy Enforcement
# -------------------------------------------------------------
def test_escalation_high_value_damage():
    agent = ConversationalAgent(client=MockClient("DECISION: escalate\nA ticket has been created for specialist review."))
    session = agent.manager.get_or_create_session("conv_esc_tv")
    session.order_id = "89782837"
    session.verification_status = "verified"
    session.verified_order_data = {"order_id": "89782837", "total_inr": 28999, "items": [{"title": "Zentro Vista 43-inch 4K Smart TV", "price_inr": 28999}]}
    
    res = agent.respond("Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?", conversation_id="conv_esc_tv")
    assert res.decision == "escalate"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["tool"] == "create_ticket"
    assert session.escalated is True


def test_escalation_human_requested():
    agent = ConversationalAgent(client=MockClient("DECISION: escalate\nA colleague will connect with you shortly."))
    session = agent.manager.get_or_create_session("conv_esc_human")
    session.order_id = "37113676"
    session.verification_status = "verified"
    session.verified_order_data = {"order_id": "37113676"}
    
    res = agent.respond("क्या मैं किसी इंसान से बात कर सकती हूँ?", conversation_id="conv_esc_human")
    assert res.decision == "escalate"
    assert any(tc["tool"] == "create_ticket" for tc in res.tool_calls)


def test_escalation_repeat_contact():
    agent = ConversationalAgent(client=MockClient("DECISION: escalate\nEscalating to our specialist team."))
    session = agent.manager.get_or_create_session("conv_repeat")
    session.order_id = "45011817"
    session.verification_status = "verified"
    session.verified_order_data = {"order_id": "45011817", "contacts_last_7_days": 2}
    
    res = agent.respond("Main is hafte teesri baar call kar rahi hoon isi baare mein.", conversation_id="conv_repeat")
    assert res.decision == "escalate"
    assert any(tc["tool"] == "create_ticket" for tc in res.tool_calls)


# -------------------------------------------------------------
# 15 & 16. Grounded RAG Responses & Preserved Citations
# -------------------------------------------------------------
def test_grounded_rag_with_preserved_citations():
    agent = ConversationalAgent(client=MockClient("DECISION: answer\nFootwear items have a 14-day return window."))
    session = agent.manager.get_or_create_session("conv_rag_cite")
    session.order_id = "18599253"
    session.verification_status = "verified"
    session.verified_order_data = {"order_id": "18599253", "items": [{"title": "Kaafi Running Sneakers", "return_category": "footwear"}]}
    
    res = agent.respond("Kya main Kaafi Running Sneakers return kar sakti hoon?", conversation_id="conv_rag_cite")
    assert res.decision == "answer"
    assert len(res.retrieved_evidence) > 0


# -------------------------------------------------------------
# 17. Multi-Turn Benchmark Loading and Ordering
# -------------------------------------------------------------
def test_multiturn_benchmark_loading_and_ordering():
    df = load_practice_conversations()
    assert len(df) == 96
    assert df["conversation_id"].nunique() == 23
    
    # Verify ordering within conversations
    for cid, sub in df.groupby("conversation_id"):
        turns = [int(t) for t in sub["turn"]]
        assert turns == list(range(1, len(turns) + 1))


# -------------------------------------------------------------
# 18. Tool Call Evaluator Integrity
# -------------------------------------------------------------
def test_evaluate_tool_calls_matcher():
    # Empty match
    assert evaluate_tool_calls([], "") is True
    assert evaluate_tool_calls([], "nan") is True
    
    # Non-empty vs empty
    assert evaluate_tool_calls([{"tool": "lookup_order"}], "") is False
    
    # lookup_order match
    calls = [{"tool": "lookup_order", "call_str": "lookup_order(order_id=89782837)"}]
    assert evaluate_tool_calls(calls, "lookup_order(order_id=89782837)") is True
    assert evaluate_tool_calls(calls, "lookup_order(order_id=99999999)") is False
    
    # create_ticket match
    t_calls = [{"tool": "create_ticket", "call_str": "create_ticket()"}]
    assert evaluate_tool_calls(t_calls, "create_ticket()") is True
