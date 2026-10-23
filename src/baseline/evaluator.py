"""Evaluation Runner for Stage 1: Prompting-Only LLM Baseline.

Processes practice/questions.csv (52 benchmark items) through BaselineAgent,
evaluates responses against dataset reference criteria after generation, and persists
results to results/stage1_baseline.csv.
"""
import os
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.config import PRACTICE_DIR, RESULTS_DIR
from src.baseline.agent import BaselineAgent, AgentResponse
from src.utils.metrics import evaluate_single_response

STAGE1_OUTPUT_CSV = RESULTS_DIR / "stage1_baseline.csv"

def load_practice_questions(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate practice/questions.csv."""
    path = csv_path or (PRACTICE_DIR / "questions.csv")
    if not path.exists():
        raise FileNotFoundError(f"Questions benchmark not found at: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(df) != 52:
        raise ValueError(f"Expected 52 practice questions, found {len(df)}")
    return df

def run_baseline_evaluation(
    agent: BaselineAgent,
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    resume: bool = True,
    use_asr_text: bool = False,
    max_questions: Optional[int] = None,
) -> pd.DataFrame:
    """Run Stage 1 evaluation across benchmark questions.
    
    Generation strictly receives only the customer question text.
    Evaluation against reference fields occurs strictly post-generation.
    """
    df_questions = load_practice_questions(csv_path)
    target_out = output_path or STAGE1_OUTPUT_CSV
    
    if max_questions:
        df_questions = df_questions.head(max_questions)
        
    existing_results: Dict[str, Dict[str, Any]] = {}
    if resume and target_out.exists():
        try:
            df_existing = pd.read_csv(target_out, dtype=str, keep_default_na=False)
            for _, row in df_existing.iterrows():
                existing_results[row["question_id"]] = row.to_dict()
            print(f"Resuming evaluation: found {len(existing_results)} already completed records.")
        except Exception as e:
            print(f"Could not read existing results, starting fresh: {e}")

    records: List[Dict[str, Any]] = []
    
    for idx, row in df_questions.iterrows():
        qid = row["id"]
        if resume and qid in existing_results:
            existing = existing_results[qid]
            if not existing.get("error"):
                records.append(existing)
                continue
            
        # Select input prompt text: strictly question or asr_text without gold evidence
        input_text = row["asr_text"] if use_asr_text and row.get("asr_text") else row["question"]
        
        # 1. Generate response strictly without reference answers or gold evidence
        response: AgentResponse = agent.respond(input_text)
        time.sleep(1.0)  # Gentle delay to respect API rate limits
        
        # 2. Evaluate response post-generation
        eval_metrics = evaluate_single_response(
            reply_text=response.reply_text,
            predicted_decision=response.decision,
            row=row.to_dict(),
        )
        
        record = {
            "question_id": qid,
            "language": row.get("language", ""),
            "intent": row.get("intent", ""),
            "input_text": input_text,
            "predicted_reply": response.reply_text,
            "predicted_decision": response.decision,
            "expected_decision": row.get("expected_decision", ""),
            "acceptable_decisions": row.get("acceptable_decisions", ""),
            "decision_match": eval_metrics["decision_match"],
            "must_include": row.get("must_include", ""),
            "must_include_passed": eval_metrics["must_include_passed"],
            "must_include_score": eval_metrics["must_include_score"],
            "missing_facts": eval_metrics["missing_facts"],
            "must_not_include": row.get("must_not_include", ""),
            "must_not_include_passed": eval_metrics["must_not_include_passed"],
            "violations": eval_metrics["violations"],
            "overall_correct": eval_metrics["overall_correct"],
            "latency_seconds": response.latency_seconds,
            "model": response.model,
            "error": response.error or "",
        }
        records.append(record)
        print(f"[{idx+1}/{len(df_questions)}] {qid} -> Decision: {response.decision} | Correct: {eval_metrics['overall_correct']}")

    # Save complete records
    df_results = pd.DataFrame(records)
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(target_out, index=False, encoding="utf-8")
    print(f"Successfully saved {len(df_results)} evaluation rows to: {target_out}")
    
    return df_results

def print_evaluation_summary(df_results: pd.DataFrame) -> None:
    """Print quantitative summary of Stage 1 baseline evaluation."""
    total = len(df_results)
    if total == 0:
        print("No evaluation records available.")
        return
        
    dec_matches = (df_results["decision_match"] == True) | (df_results["decision_match"] == "True")
    inc_passed = (df_results["must_include_passed"] == True) | (df_results["must_include_passed"] == "True")
    not_inc_passed = (df_results["must_not_include_passed"] == True) | (df_results["must_not_include_passed"] == "True")
    overall_correct = (df_results["overall_correct"] == True) | (df_results["overall_correct"] == "True")
    errors = df_results[df_results["error"] != ""]
    
    print("\n" + "="*50)
    print("STAGE 1 PROMPTING-ONLY BASELINE: EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Questions Evaluated: {total}")
    print(f"Decision Accuracy:        {dec_matches.sum()}/{total} ({dec_matches.mean()*100:.1f}%)")
    print(f"Must-Include Fact Recall: {inc_passed.sum()}/{total} ({inc_passed.mean()*100:.1f}%)")
    print(f"Must-Not-Include Passed:  {not_inc_passed.sum()}/{total} ({not_inc_passed.mean()*100:.1f}%)")
    print(f"Overall Grounded Match:   {overall_correct.sum()}/{total} ({overall_correct.mean()*100:.1f}%)")
    if len(errors) > 0:
        print(f"Errors / Failures:        {len(errors)}/{total}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 1 Prompting Baseline Evaluation.")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all questions.")
    parser.add_argument("--asr", action="store_true", help="Use asr_text instead of clean question text.")
    args = parser.parse_args()
    
    try:
        agent = BaselineAgent()
        print(f"Initialized BaselineAgent with model: {agent.model}")
        results = run_baseline_evaluation(agent, resume=not args.force, use_asr_text=args.asr)
        print_evaluation_summary(results)
    except ValueError as e:
        print(f"\n[BLOCKED] Authentication Error: {e}")
        print("Configure GEMINI_API_KEY in your local environment or .env file.")
        print("Live evaluation remains PENDING until credentials are provided.")
