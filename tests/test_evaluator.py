"""Unit tests for Stage 1 Evaluator and Metrics (Milestone 1)."""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from pathlib import Path

from src.utils.metrics import (
    evaluate_decision, evaluate_must_include, evaluate_must_not_include,
    evaluate_single_response
)
from src.baseline.evaluator import run_baseline_evaluation, load_practice_questions
from src.baseline.agent import BaselineAgent

def test_load_practice_questions_count():
    df = load_practice_questions()
    assert len(df) == 52
    assert "question" in df.columns
    assert "expected_decision" in df.columns
    assert "must_include" in df.columns
    assert "must_not_include" in df.columns

def test_metrics_decision_evaluation():
    assert evaluate_decision("answer", "answer", "answer ; clarify") is True
    assert evaluate_decision("clarify", "answer", "answer ; clarify") is True
    assert evaluate_decision("refuse", "answer", "answer ; clarify") is False
    assert evaluate_decision("", "answer", "answer") is False

def test_metrics_must_include_evaluation():
    spec = "12 ; My Orders / app"
    # Case with both facts present
    reply_good = "Yes, 12 months warranty is available in the app under My Orders."
    passed, score, missing = evaluate_must_include(reply_good, spec)
    assert passed is True
    assert score == 1.0
    assert missing == []

    # Case with only one fact present (12 months present, app/My Orders missing)
    reply_partial = "Yes, 12 months warranty is available from the manufacturer."
    passed, score, missing = evaluate_must_include(reply_partial, spec)
    assert passed is False
    assert score == 0.5
    assert len(missing) == 1

def test_metrics_must_not_include_evaluation():
    spec = "5 to 7 days ; 5-7 days"
    # Clean reply
    reply_clean = "Refunds to wallet take within 2 hours."
    passed, violations = evaluate_must_not_include(reply_clean, spec)
    assert passed is True
    assert violations == []

    # Violating reply
    reply_bad = "Refunds take 5 to 7 days to reach your wallet."
    passed, violations = evaluate_must_not_include(reply_bad, spec)
    assert passed is False
    assert len(violations) == 1
    assert "5 to 7 days" in violations

def test_evaluator_pipeline_mock_run(tmp_path):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = (
        "Hello! I am ZentroMart's AI support assistant. Yes, 12 months warranty is in My Orders.\n"
        "DECISION: answer"
    )
    mock_client.models.generate_content.return_value = mock_resp
    agent = BaselineAgent(api_key="mock_key", client=mock_client)

    test_out = tmp_path / "test_stage1_output.csv"
    results = run_baseline_evaluation(
        agent=agent,
        output_path=test_out,
        resume=False,
        max_questions=2
    )

    assert len(results) == 2
    assert test_out.exists()
    assert "question_id" in results.columns
    assert "predicted_decision" in results.columns
    assert "overall_correct" in results.columns
