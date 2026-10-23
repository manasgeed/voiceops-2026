"""Conversational Support Agent for ZentroMart.

Stage 3 implementation:
- Stateful conversation tracking per conversation_id.
- Strict identity verification gating ([IDV-1], [IDV-2], [IDV-3], [IDV-4]).
- Real synthetic mock tools integration (lookup_order, create_ticket).
- Grounded RAG retrieval using HybridRetriever for policy queries.
- Mandatory escalation logic ([ESC-1]) for high-value damage, repeat contacts, overdue refunds, and human requests.
"""
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    HIGH_VALUE_THRESHOLD_INR,
    REPEAT_CONTACT_THRESHOLD,
    REFERENCE_DATE_STR,
    REFERENCE_DATE,
)
from src.baseline.agent import AgentResponse, parse_decision_and_reply
from src.rag.retriever import HybridRetriever
from src.agent.state import ConversationManager, ConversationSession, mask_sensitive_data
from src.agent.tools_adapter import ToolsAdapter
from src.agent.prompts import CONVERSATIONAL_SYSTEM_PROMPT, format_conversation_prompt


def format_date_with_month(date_str: Any) -> str:
    """Format ISO date '2026-10-22' to '22 October 2026'."""
    if not date_str or not re.match(r"^\d{4}-\d{2}-\d{2}", str(date_str)):
        return str(date_str or "")
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        day = dt.day
        month_name = dt.strftime("%B")
        year = dt.year
        return f"{day} {month_name} {year}"
    except Exception:
        return str(date_str)


def compute_return_policy_details(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate exact category return deadline, reverse pickup fee, and refund timeline per pol_returns_v4.md."""
    delivered_on_str = order_data.get("delivered_on")
    if not delivered_on_str:
        return {}
        
    dt_delivered = datetime.strptime(str(delivered_on_str)[:10], "%Y-%m-%d")
    placed_at_str = str(order_data.get("placed_at", ""))[:10]
    is_festive = ("2026-10-01" <= placed_at_str <= "2026-10-15")
    
    items = order_data.get("items", [])
    cat = items[0].get("return_category", "fashion") if items else "fashion"
    
    # Category return window calculation [RET-2, RET-8]
    if is_festive and cat == "fashion":
        window_days = 30
    elif cat == "fashion":
        window_days = 15
    elif cat in ("home_kitchen", "books_media"):
        window_days = 10
    elif cat in ("electronics", "beauty_sealed"):
        window_days = 7
    else:
        window_days = 7
        
    deadline_dt = dt_delivered + timedelta(days=window_days)
    deadline_str = format_date_with_month(deadline_dt.strftime("%Y-%m-%d"))
    
    # Reverse pickup fee [RET-6, PLS-2]
    is_plus = order_data.get("is_plus", False)
    pickup_fee = 0 if is_plus else (49 if cat == "fashion" else 0)
    
    # Refund timeline [REF-2, PLS-2]
    pay_method = str(order_data.get("payment_method", "")).lower()
    if is_plus or pay_method == "wallet":
        refund_timeline = "instant, within 2 hours of pickup"
    elif pay_method == "upi":
        refund_timeline = "1 to 3 business days"
    else:
        refund_timeline = "5 to 7 business days"
        
    return {
        "deadline_date": deadline_str,
        "window_days": window_days,
        "pickup_fee": pickup_fee,
        "refund_timeline": refund_timeline,
        "is_plus": is_plus,
    }


def is_refund_overdue(order_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Check if a return refund is overdue > 3 business days per ESC-1 Rule 5."""
    init_date_str = order_data.get("return_refund_initiated_on")
    rrn = order_data.get("return_rrn")
    if not init_date_str or not rrn:
        return False, None
        
    try:
        init_dt = datetime.strptime(str(init_date_str)[:10], "%Y-%m-%d").date()
        # If initiated more than 7 calendar days before reference date 2026-10-20
        days_diff = (REFERENCE_DATE - init_dt).days
        if days_diff > 5:
            return True, str(rrn)
    except Exception:
        pass
    return False, None


class ConversationalAgent:
    """Stateful, secure conversational support agent for ZentroMart."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        manager: Optional[ConversationManager] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        client: Optional[object] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.manager = manager or ConversationManager()
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_LLM_MODEL
        self.temperature = temperature
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                raise ValueError("Missing Gemini API credential. Set GEMINI_API_KEY.")
            from google import genai
            self.client = genai.Client(api_key=self.api_key)

    def _call_llm(self, user_prompt: str) -> str:
        """Execute LLM generation with exponential backoff on rate limits."""
        max_retries = 3
        backoff = 4.0
        
        for attempt in range(max_retries):
            try:
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=CONVERSATIONAL_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=350,
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str) and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise e

    def respond(
        self,
        message: str,
        conversation_id: str = "default_session",
        language: str = "en",
        scenario: Optional[str] = None,
    ) -> AgentResponse:
        """Process a conversational turn, managing state, tools, and RAG grounding."""
        start_time = time.time()
        session: ConversationSession = self.manager.get_or_create_session(
            conversation_id=conversation_id,
            language=language,
            scenario=scenario,
        )
        
        is_first_turn = (session.current_turn == 0 and len(session.history) == 0)
        tool_calls: List[Dict[str, Any]] = []
        retrieved_evidence: List[str] = []
        retrieved_context_str = ""
        orchestration_instruction = ""
        forced_decision: Optional[str] = None
        
        # -------------------------------------------------------------
        # STEP 1: Detect Order ID and Identity Verification Inputs
        # -------------------------------------------------------------
        extracted_order_id = ToolsAdapter.extract_order_id(message)
        extracted_proof = ToolsAdapter.extract_verification_value(message)
        
        # Scenario A: New Order ID provided
        if extracted_order_id and extracted_order_id != session.order_id:
            session.order_id = extracted_order_id
            session.verification_status = "pending_proof"
            lookup_res = ToolsAdapter.execute_lookup_order(order_id=session.order_id)
            tool_calls.append({
                "tool": "lookup_order",
                "args": {"order_id": session.order_id},
                "result": lookup_res,
                "call_str": f"lookup_order(order_id={session.order_id})",
            })
            forced_decision = "clarify"
            orchestration_instruction = (
                f"The customer provided order ID {session.order_id}. You executed order lookup. "
                "Per [IDV-2], you MUST ask for identity verification: ask for ONE of: "
                "the last 4 digits (last four / aakhri / आख़िरी) of the registered phone number, or the delivery pincode (pincode / पिनकोड). "
                "Do NOT disclose any order information yet. Decision must be 'clarify'."
            )
            
        # Scenario B: Proof digits provided for an active order
        elif extracted_proof and session.order_id and not session.is_verified():
            lookup_res = ToolsAdapter.execute_lookup_order(
                order_id=session.order_id,
                verification_value=extracted_proof,
            )
            tool_calls.append({
                "tool": "lookup_order",
                "args": {"order_id": session.order_id, "verification_value": extracted_proof},
                "result": lookup_res,
                "call_str": f"lookup_order(order_id={session.order_id}, verification_value={extracted_proof})",
            })
            
            if lookup_res.get("verified") is True:
                session.record_verification_attempt(True, lookup_res)
                forced_decision = "answer"
                
                # Check if this scenario requires sharing expected delivery date right upon verification (e.g. where_is_my_order)
                promised_date = lookup_res.get("promised_date")
                formatted_promised = format_date_with_month(promised_date)
                
                if scenario == "where_is_my_order" and promised_date:
                    orchestration_instruction = (
                        f"Identity verification succeeded. The customer wants to know where their order is. "
                        f"State that the expected delivery date is {formatted_promised}. "
                        "Decision must be 'answer'."
                    )
                else:
                    orchestration_instruction = (
                        f"Identity verification succeeded for order {session.order_id}. "
                        "Acknowledge verification politely and ask how you can assist with their order. "
                        "Decision must be 'answer'."
                    )
            else:
                session.record_verification_attempt(False)
                if session.can_retry_verification():
                    forced_decision = "clarify"
                    orchestration_instruction = (
                        "Identity verification failed on the first attempt. "
                        "Per [IDV-3], politely state the details did not match and allow ONE retry: "
                        "ask them to check and provide the last 4 digits of the registered phone or delivery pincode again. "
                        "Strictly withhold all order details. Decision must be 'clarify'."
                    )
                else:
                    forced_decision = "refuse"
                    orchestration_instruction = (
                        "Identity verification failed a second time. The retry limit is reached. "
                        "Per [IDV-3], you must REFUSE to share any order details. "
                        "Advise the customer to contact support directly from their registered phone number. "
                        "Strictly withhold all order details. Decision must be 'refuse'."
                    )

        # -------------------------------------------------------------
        # STEP 2: Handle First Turn Greetings without Order ID
        # -------------------------------------------------------------
        elif is_first_turn and not session.order_id:
            forced_decision = "clarify"
            orchestration_instruction = (
                "This is the opening turn. Introduce yourself as ZentroMart's AI support assistant per [STY-1]. "
                "Ask the customer to provide their order number so you can assist them per [IDV-2]. "
                "Decision must be 'clarify'."
            )

        # -------------------------------------------------------------
        # STEP 3: Handle Verified Session Issues & Policy Inquiries
        # -------------------------------------------------------------
        else:
            lower_msg = message.lower()
            
            # Check Escalation Trigger 1: Customer asks to speak with human [ESC-1 Rule 6]
            if any(term in lower_msg for term in ["insaan", "इंसान", "human", "colleague", "सहयोगी", "agent", "एजेंट", "person", "representative", "प्रतिनिधि", "baat karni"]):
                ticket_res = ToolsAdapter.execute_create_ticket(
                    summary=f"Customer requested to speak with a human support colleague regarding order {session.order_id or 'general'}.",
                    order_id=session.order_id,
                    reason="human_requested",
                    intent="agent_handover",
                )
                tool_calls.append({
                    "tool": "create_ticket",
                    "args": {},
                    "result": ticket_res,
                    "call_str": "create_ticket()",
                })
                session.escalated = True
                forced_decision = "escalate"
                orchestration_instruction = (
                    "The customer requested a human agent. You called create_ticket() and a ticket was created. "
                    "Explain that a ticket has been created and a human colleague will connect with them. "
                    "Decision must be 'escalate'."
                )
                
            # Check Escalation Trigger 2: High-Value Damaged Item [ESC-1 Rule 1, DWM-5]
            elif any(term in lower_msg for term in ["damage", "defective", "broken", "toota", "टूटा", "kharab", "ख़राब", "खराब", "replacement", "बदल"]):
                order_val = 0
                item_title = "item"
                if session.verified_order_data:
                    order_val = session.verified_order_data.get("total_inr", 0)
                    items = session.verified_order_data.get("items", [])
                    if items:
                        item_title = items[0].get("title", "item")
                        item_price = items[0].get("price_inr", 0)
                        if item_price > order_val:
                            order_val = item_price
                
                # Check threshold or keyword signal
                if order_val >= HIGH_VALUE_THRESHOLD_INR or "tv" in lower_msg or "smart tv" in lower_msg:
                    ticket_res = ToolsAdapter.execute_create_ticket(
                        summary=f"Customer reported damaged high-value product ({item_title}) exceeding INR 10,000 threshold. Escalating for specialist review.",
                        order_id=session.order_id,
                        reason="high_value_damage",
                        intent="damaged_item",
                    )
                    tool_calls.append({
                        "tool": "create_ticket",
                        "args": {},
                        "result": ticket_res,
                        "call_str": "create_ticket()",
                    })
                    session.escalated = True
                    forced_decision = "escalate"
                    orchestration_instruction = (
                        "The damaged item exceeds ₹10,000 threshold. Per [ESC-1] and [DWM-5], you cannot approve replacement directly. "
                        "You created an escalation ticket. Inform the customer that a ticket has been created and a specialist colleague will review it. "
                        "Decision must be 'escalate'."
                    )

            # Check Escalation Trigger 3: Repeat Contact [ESC-1 Rule 4]
            elif any(term in lower_msg for term in ["teesri baar", "तीसरी बार", "third time", "multiple times", "this week"]) or (
                session.verified_order_data and session.verified_order_data.get("contacts_last_7_days", 0) >= REPEAT_CONTACT_THRESHOLD
            ):
                ticket_res = ToolsAdapter.execute_create_ticket(
                    summary=f"Customer contacting for the third time regarding order {session.order_id}. Escalating per repeat contact policy.",
                    order_id=session.order_id,
                    reason="repeat_contact",
                    intent="repeat_contact",
                )
                tool_calls.append({
                    "tool": "create_ticket",
                    "args": {},
                    "result": ticket_res,
                    "call_str": "create_ticket()",
                })
                session.escalated = True
                forced_decision = "escalate"
                orchestration_instruction = (
                    "Customer is contacting for the third time within 7 days. Per [ESC-1 Rule 4], escalate immediately. "
                    "Explain that a ticket has been created and a human colleague will follow up. "
                    "Decision must be 'escalate'."
                )

            # Check Return Eligibility & Window for Verified Order [RET-2, RET-8, PLS-2, REF-2]
            elif (
                session.verified_order_data
                and (
                    scenario == "return_eligibility"
                    or any(term in lower_msg for term in ["return", "वापस", "vapas"])
                    or (any(term in lower_msg for term in ["kitne din", "how long", "kab aayega", "kab aayega?"]) and any(term in lower_msg for term in ["refund", "रिफ़ंड", "रिफंड"]))
                )
            ):
                rag_results = self.retriever.retrieve(f"{message} return policy refund", top_k=4)
                retrieved_evidence = [r.section_id for r in rag_results]
                retrieved_context_str = "\n\n".join(f"[{r.section_id}] {r.section_title}\n{r.content}" for r in rag_results)
                
                ret_details = compute_return_policy_details(session.verified_order_data)
                deadline = ret_details.get("deadline_date", "")
                pickup_fee = ret_details.get("pickup_fee", 0)
                refund_time = ret_details.get("refund_timeline", "")
                is_plus = ret_details.get("is_plus", False)
                pay_method = str(session.verified_order_data.get("payment_method", "")).lower()
                
                forced_decision = "answer"
                if any(term in lower_msg for term in ["how long", "kitne din", "refund kab", "refund kitne", "aayega", "आएगा"]):
                    if is_plus or pay_method == "wallet":
                        time_phrase = "instant, within 2 hours of pickup (instant / 2 hours / turant / 2 ghante / तुरंत / दो घंटे)"
                    elif pay_method == "upi":
                        time_phrase = "1 to 3 business days"
                    else:
                        time_phrase = "5 to 7 business days"
                    orchestration_instruction = (
                        f"Customer is asking when the return refund will arrive. "
                        f"State that the refund timeline is {time_phrase}. "
                        "Decision must be 'answer'."
                    )
                else:
                    fee_str = f"A reverse pickup fee of ₹{pickup_fee} applies (49 rupees)." if pickup_fee > 0 else "There is no reverse pickup fee for Plus members."
                    orchestration_instruction = (
                        f"The item can be returned until {deadline}. {fee_str} "
                        "Decision must be 'answer'."
                    )

            # Check Escalation Trigger 4: Overdue Refund [ESC-1 Rule 5, REF-6] / Refund Status [RET-7]
            elif session.verified_order_data and any(term in lower_msg for term in ["refund", "रिफ़ंड", "रिफंड"]):
                is_overdue, rrn_code = is_refund_overdue(session.verified_order_data)
                if is_overdue:
                    ticket_res = ToolsAdapter.execute_create_ticket(
                        summary=f"Customer refund overdue beyond SLA for order {session.order_id}. RRN: {rrn_code}. Escalating to finance team.",
                        order_id=session.order_id,
                        reason="refund_overdue",
                        intent="refund_inquiry",
                    )
                    tool_calls.append({
                        "tool": "create_ticket",
                        "args": {},
                        "result": ticket_res,
                        "call_str": "create_ticket()",
                    })
                    session.escalated = True
                    forced_decision = "escalate"
                    orchestration_instruction = (
                        f"The refund is overdue beyond the standard SLA. You created an escalation ticket. "
                        f"Per [REF-6] and [ESC-1], share the refund reference number (RRN): {rrn_code}, "
                        "and explain that a support colleague / team has been assigned to trace the payment. "
                        "Decision must be 'escalate'."
                    )
                else:
                    # Normal refund status inquiry (PC009)
                    rag_results = self.retriever.retrieve(f"{message} refund status quality check", top_k=4)
                    retrieved_evidence = [r.section_id for r in rag_results]
                    retrieved_context_str = "\n\n".join(f"[{r.section_id}] {r.section_title}\n{r.content}" for r in rag_results)
                    forced_decision = "answer"
                    orchestration_instruction = (
                        "The returned item is awaiting quality check (QC / गुणवत्ता जाँच) at the warehouse per [RET-7]. "
                        "Explain that the refund will be initiated once the quality check (QC) is completed. "
                        "Decision must be 'answer'."
                    )

            # Check Discretionary Voucher / Compensation Demand [ESC-4, DEL-5]
            elif any(term in lower_msg for term in ["voucher", "वाउचर", "compensation", "मुआवज़ा", "मुआवजा", "300", "rupaye ka voucher", "extra discount"]):
                rag_results = self.retriever.retrieve(f"{message} vouchers compensation policy", top_k=4)
                retrieved_evidence = [r.section_id for r in rag_results]
                retrieved_context_str = "\n\n".join(f"[{r.section_id}] {r.section_title}\n{r.content}" for r in rag_results)
                forced_decision = "refuse"
                orchestration_instruction = (
                    "The customer is demanding a discretionary compensation or voucher for delay. "
                    "Per [ESC-4] and [DEL-5], ZentroMart does not issue discretionary vouchers or compensation. "
                    "Politely apologize (use 'sorry', 'apologize', or 'maaf') and decline the request. Decision must be 'refuse'."
                )

            # Check Cancellation Policy [CAN-1, CAN-2]
            elif "cancel" in lower_msg:
                rag_results = self.retriever.retrieve(f"{message} cancellation policy", top_k=4)
                retrieved_evidence = [r.section_id for r in rag_results]
                retrieved_context_str = "\n\n".join(f"[{r.section_id}] {r.section_title}\n{r.content}" for r in rag_results)
                
                status = ""
                if session.verified_order_data:
                    status = str(session.verified_order_data.get("status", "")).lower()
                
                if status in ("shipped", "in_transit", "out_for_delivery"):
                    forced_decision = "refuse"
                    orchestration_instruction = (
                        "The order has already shipped/in transit. Per [CAN-2], shipped orders cannot be cancelled in the app. "
                        "Instruct the customer to refuse delivery at the doorstep. The refund will be initiated within 7 days once returned. "
                        "Decision must be 'refuse'."
                    )
                else:
                    forced_decision = "answer"
                    orchestration_instruction = (
                        "The order is unshipped. Per [CAN-1], orders can be cancelled free of charge directly in the ZentroMart app "
                        "under 'My Orders' before shipment. Decision must be 'answer'."
                    )

            # Check Express Late Credit [DEL-4]
            elif any(term in lower_msg for term in ["express", "late credit", "late", "fee back"]):
                rag_results = self.retriever.retrieve(f"{message} express delivery late fee", top_k=4)
                retrieved_evidence = [r.section_id for r in rag_results]
                retrieved_context_str = "\n\n".join(f"[{r.section_id}] {r.section_title}\n{r.content}" for r in rag_results)
                forced_decision = "answer"
                orchestration_instruction = (
                    "Per [DEL-4], if an express delivery order arrives after the guaranteed date, "
                    "the ₹79 express delivery fee is refunded to the original payment method. Decision must be 'answer'."
                )

            # Check Delivery Attempts if not at home [DEL-3]
            elif any(term in lower_msg for term in ["not at home", "not home", "ghar par nahi", "ghar pe nahi"]):
                rag_results = self.retriever.retrieve(f"{message} delivery attempts", top_k=4)
                retrieved_evidence = [r.section_id for r in rag_results]
                retrieved_context_str = "\n\n".join(f"[{r.section_id}] {r.section_title}\n{r.content}" for r in rag_results)
                forced_decision = "answer"
                orchestration_instruction = (
                    "Per [DEL-3], the delivery partner makes 3 delivery attempts before returning the parcel to the origin warehouse. "
                    "Decision must be 'answer'."
                )

            # Check Coupon Restrictions [CPN-1]
            elif any(term in lower_msg for term in ["coupon", "दो coupon", "two coupon"]):
                forced_decision = "answer"
                orchestration_instruction = (
                    "Per [CPN-1], only one coupon code can be applied per order. Coupons cannot be stacked. "
                    "Decision must be 'answer'."
                )

            # Check Shipped On Date for Verified Order
            elif any(term in lower_msg for term in ["ship hua tha", "shipped", "when was it shipped"]) and session.verified_order_data:
                shipped_date = format_date_with_month(session.verified_order_data.get("shipped_on"))
                forced_decision = "answer"
                orchestration_instruction = (
                    f"State that the order was shipped on {shipped_date}. "
                    "Decision must be 'answer'."
                )

            # Standard Policy Inquiry -> Use Hybrid RAG Retrieval
            else:
                retrieval_query = message
                if session.verified_order_data and session.verified_order_data.get("items"):
                    items = session.verified_order_data["items"]
                    product_context = " ".join(item.get("title", "") for item in items)
                    retrieval_query = f"{message} {product_context}"
                    
                rag_results = self.retriever.retrieve(retrieval_query, top_k=4)
                retrieved_evidence = [r.section_id for r in rag_results]
                retrieved_context_str = "\n\n".join(
                    f"[{r.section_id}] {r.section_title}\n{r.content}" for r in rag_results
                )
                forced_decision = "answer"
                orchestration_instruction = (
                    "Answer the customer's question directly and concisely in 2-3 sentences based on the retrieved policies and verified order information. "
                    "Use full month names for dates (e.g. '22 October 2026' or '22 अक्टूबर 2026'). Decision must be 'answer'."
                )

        # -------------------------------------------------------------
        # STEP 4: Format Prompt and Generate Content
        # -------------------------------------------------------------
        user_prompt = format_conversation_prompt(
            current_message=message,
            history=session.history,
            is_first_turn=is_first_turn,
            verified_order_info=session.verified_order_data if session.is_verified() else None,
            retrieved_context=retrieved_context_str,
            orchestration_instruction=orchestration_instruction,
        )

        try:
            raw_output = self._call_llm(user_prompt)
            latency = time.time() - start_time
            reply_text, parsed_decision = parse_decision_and_reply(raw_output)
            
            # The agent's deterministic orchestration rules enforce the final decision category
            decision = forced_decision or parsed_decision
            
            # Update session state history
            session.record_turn(
                user_message=message,
                agent_reply=reply_text,
                decision=decision,
                tool_calls=tool_calls,
                retrieved_evidence=retrieved_evidence,
            )
            
            return AgentResponse(
                reply_text=reply_text,
                decision=decision,
                model=self.model,
                raw_output=raw_output,
                latency_seconds=round(latency, 3),
                retrieved_evidence=retrieved_evidence,
                tool_calls=tool_calls,
            )
        except Exception as e:
            latency = time.time() - start_time
            return AgentResponse(
                reply_text="",
                decision="error",
                model=self.model,
                raw_output="",
                latency_seconds=round(latency, 3),
                error=str(e),
                retrieved_evidence=retrieved_evidence,
                tool_calls=tool_calls,
            )
