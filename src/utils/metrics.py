"""Evaluation metric functions conforming to DATA_DICTIONARY.md rules.

Metrics calculated:
- Decision accuracy (matching expected_decision or any of acceptable_decisions).
- must_include factual recall (all semicolon-delimited facts required; slash-delimited alternatives are equivalent).
- must_not_include constraint adherence (zero forbidden phrases present).
- Overall answer correctness.
"""
import re
from typing import Dict, Any, List

def normalize_text(text: str) -> str:
    """Normalize text for evaluation: lowercase, collapse whitespace, strip punctuation."""
    if not text:
        return ""
    # Collapse multiple whitespaces and lowercase
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    return cleaned

def evaluate_decision(predicted_decision: str, expected_decision: str, acceptable_decisions: str) -> bool:
    """Check if predicted decision matches expected or acceptable decisions."""
    if not predicted_decision:
        return False
    pred = predicted_decision.strip().lower()
    
    candidates = set()
    if expected_decision and str(expected_decision).strip() != "nan":
        candidates.add(expected_decision.strip().lower())
    if acceptable_decisions and str(acceptable_decisions).strip() != "nan":
        for d in acceptable_decisions.split(";"):
            candidates.add(d.strip().lower())
            
    return pred in candidates

def evaluate_must_include(reply_text: str, must_include_spec: str) -> tuple[bool, float, List[str]]:
    """Evaluate must_include facts.
    
    Format: Facts separated by ' ; '. Within a fact, alternatives separated by ' / '.
    Returns (all_passed, ratio_passed, missing_facts).
    """
    if not must_include_spec or str(must_include_spec).strip() in ("", "nan"):
        return True, 1.0, []
        
    norm_reply = normalize_text(reply_text)
    facts = [f.strip() for f in must_include_spec.split(";") if f.strip()]
    if not facts:
        return True, 1.0, []
        
    met_count = 0
    missing = []
    
    for fact in facts:
        alternatives = [a.strip() for a in fact.split("/") if a.strip()]
        matched = False
        for alt in alternatives:
            norm_alt = normalize_text(alt)
            # Support word boundary match or substring for numbers/dates
            if norm_alt in norm_reply:
                matched = True
                break
        if matched:
            met_count += 1
        else:
            missing.append(fact)
            
    ratio = met_count / len(facts)
    all_passed = (met_count == len(facts))
    return all_passed, round(ratio, 3), missing

def evaluate_must_not_include(reply_text: str, must_not_include_spec: str) -> tuple[bool, List[str]]:
    """Evaluate must_not_include constraints.
    
    Format: Forbidden phrases separated by ' ; '.
    Returns (passed, list_of_violations).
    """
    if not must_not_include_spec or str(must_not_include_spec).strip() in ("", "nan"):
        return True, []
        
    norm_reply = normalize_text(reply_text)
    forbidden = [f.strip() for f in must_not_include_spec.split(";") if f.strip()]
    if not forbidden:
        return True, []
        
    violations = []
    for phrase in forbidden:
        norm_phrase = normalize_text(phrase)
        if norm_phrase in norm_reply:
            violations.append(phrase)
            
    passed = (len(violations) == 0)
    return passed, violations

def evaluate_single_response(
    reply_text: str,
    predicted_decision: str,
    row: Dict[str, Any]
) -> Dict[str, Any]:
    """Score a single generated response against practice/questions.csv row fields."""
    exp_decision = str(row.get("expected_decision", ""))
    acc_decisions = str(row.get("acceptable_decisions", ""))
    must_inc = str(row.get("must_include", ""))
    must_not_inc = str(row.get("must_not_include", ""))
    
    dec_match = evaluate_decision(predicted_decision, exp_decision, acc_decisions)
    inc_passed, inc_score, missing_facts = evaluate_must_include(reply_text, must_inc)
    not_inc_passed, violations = evaluate_must_not_include(reply_text, must_not_inc)
    
    # An answer is strictly correct if decision matches, all required facts are present,
    # and no forbidden claims are made
    overall_correct = bool(dec_match and inc_passed and not_inc_passed)
    
    return {
        "decision_match": dec_match,
        "must_include_passed": inc_passed,
        "must_include_score": inc_score,
        "missing_facts": " ; ".join(missing_facts) if missing_facts else "",
        "must_not_include_passed": not_inc_passed,
        "violations": " ; ".join(violations) if violations else "",
        "overall_correct": overall_correct,
    }
