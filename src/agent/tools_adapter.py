"""Tools Adapter for ZentroMart Support Agent.

Safely wraps the official synthetic offline tools from tools/tools.py:
1. lookup_order(order_id, verification_value=None)
2. create_ticket(summary, order_id=None, reason=None, intent=None)

Enforces verification gating, sanitization, and summary constraints.
Never leaks verification secrets or circumvents tool boundaries.
"""
import sys
import re
from typing import Dict, Any, Optional

from src.config import TOOLS_DIR, MIN_TICKET_SUMMARY_LENGTH
from src.agent.state import mask_sensitive_data

# Ensure official tools directory is accessible
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from tools import lookup_order as _raw_lookup_order  # type: ignore
    from tools import create_ticket as _raw_create_ticket  # type: ignore
except ImportError as e:
    raise ImportError(f"Could not import official tools from {TOOLS_DIR}: {e}")


class ToolsAdapter:
    """Secure adapter around official mock tools."""

    @staticmethod
    def extract_order_id(text: str) -> Optional[str]:
        """Extract 8-digit order number from user text."""
        if not text:
            return None
        # Remove whitespace between digits if formatted as '8978 2837'
        clean = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
        match = re.search(r"\b(\d{8})\b", clean)
        if match:
            return match.group(1)
        # Fallback: check 8 continuous digits in any token
        digits = re.findall(r"\d+", text)
        for d in digits:
            if len(d) == 8:
                return d
        if len("".join(digits)) == 8:
            return "".join(digits)
        return None

    @staticmethod
    def extract_verification_value(text: str) -> Optional[str]:
        """Extract 4-digit phone suffix or 6-digit pincode from user text."""
        if not text:
            return None
        # Look for explicit 6-digit pincode
        pincode_match = re.search(r"\b(\d{6})\b", text)
        if pincode_match:
            return pincode_match.group(1)
        # Look for explicit 4-digit phone suffix
        four_digit_match = re.search(r"\b(\d{4})\b", text)
        if four_digit_match:
            return four_digit_match.group(1)
        # Check if digits are provided directly
        digits = re.sub(r"\D", "", text)
        if len(digits) in (4, 6):
            return digits
        return None

    @staticmethod
    def execute_lookup_order(order_id: str, verification_value: Optional[str] = None) -> Dict[str, Any]:
        """Call the official lookup_order tool strictly via its defined signature.
        
        Never queries raw CSV files or circumvents identity verification.
        """
        oid = "".join(ch for ch in str(order_id) if "0" <= ch <= "9")
        proof = "".join(ch for ch in str(verification_value) if "0" <= ch <= "9") if verification_value else None
        
        result = _raw_lookup_order(order_id=oid, verification_value=proof)
        return result

    @staticmethod
    def execute_create_ticket(
        summary: str,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the official create_ticket tool enforcing minimum summary length.
        
        Ensures summary is >= 20 characters and contains no raw verification secrets.
        """
        safe_summary = mask_sensitive_data(summary.strip())
        if len(safe_summary) < MIN_TICKET_SUMMARY_LENGTH:
            safe_summary = f"{safe_summary} | Checks performed and escalated per ZentroMart policy."
            
        result = _raw_create_ticket(
            summary=safe_summary,
            order_id=order_id or "",
            reason=reason or "",
            intent=intent or "",
        )
        return result
