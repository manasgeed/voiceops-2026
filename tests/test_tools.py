import sys
from pathlib import Path
from src.config import TOOLS_DIR

# Add tools dir to sys.path so we import the official mock tools
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools import lookup_order, create_ticket  # type: ignore

def test_lookup_order_unverified_withholds_details():
    res = lookup_order("8978 2837")
    assert res.get("verification_required") is True
    assert "status" not in res
    assert "items" not in res
    assert "total_inr" not in res
    assert "customer_first_name" not in res

def test_lookup_order_verification_with_phone_last4():
    # Order 89782837 belongs to C100185 whose phone ends in 4533
    res = lookup_order("8978 2837", verification_value="4533")
    assert res.get("found") is True
    assert res.get("verified") is True
    assert "status" in res
    assert "items" in res
    assert "total_inr" in res
    assert res.get("customer_first_name") is not None

def test_lookup_order_verification_with_pincode():
    # Order 89782837 delivery pincode is 781006
    res = lookup_order("8978 2837", verification_value="781006")
    assert res.get("found") is True
    assert res.get("verified") is True
    assert "status" in res

def test_lookup_order_invalid_verification_fails():
    res = lookup_order("8978 2837", verification_value="9999")
    assert res.get("verified") is False
    assert res.get("verification_required") is True
    assert "status" not in res
    assert "items" not in res

def test_create_ticket_min_summary_length():
    # Less than 20 chars must fail
    res_short = create_ticket(summary="Short summary")
    assert res_short.get("created") is False
    assert "error" in res_short
    assert "summary too short" in res_short["error"]

    # At least 20 chars must succeed
    valid_summary = "Customer reported damaged television exceeding INR 10,000 threshold."
    assert len(valid_summary) >= 20
    res_valid = create_ticket(summary=valid_summary, order_id="89782837", reason="high_value_damage")
    assert res_valid.get("created") is True
    assert res_valid.get("ticket_id") is not None
    assert str(res_valid["ticket_id"]).startswith("TKT")
