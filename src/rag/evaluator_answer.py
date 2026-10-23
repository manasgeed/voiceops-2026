"""Stage 2 Grounded RAG Answer Evaluator.

Processes practice/questions.csv through RAGAgent, evaluates responses against
dataset reference criteria (decision_match, must_include, must_not_include),
and writes outputs to results/stage2_rag.csv.
"""
import os
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.config import PRACTICE_DIR, RESULTS_DIR
from src.rag.agent import RAGAgent, AgentResponse
from src.utils.metrics import evaluate_single_response
from src.baseline.evaluator import load_practice_questions, print_evaluation_summary

STAGE2_OUTPUT_CSV = RESULTS_DIR / "stage2_rag.csv"

def run_rag_answer_evaluation(
    agent: RAGAgent,
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    resume: bool = True,
    use_asr_text: bool = False,
    max_questions: Optional[int] = None,
) -> pd.DataFrame:
    """Run Stage 2 evaluation across all 52 benchmark questions."""
    df_questions = load_practice_questions(csv_path)
    target_out = output_path or STAGE2_OUTPUT_CSV
    
    if max_questions:
        df_questions = df_questions.head(max_questions)
        
    existing_results: Dict[str, Dict[str, Any]] = {}
    if resume and target_out.exists():
        try:
            df_existing = pd.read_csv(target_out, dtype=str, keep_default_na=False)
            for _, row in df_existing.iterrows():
                existing_results[row["question_id"]] = row.to_dict()
            print(f"Resuming RAG evaluation: found {len(existing_results)} already completed records.")
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
            
        input_text = row["asr_text"] if use_asr_text and row.get("asr_text") else row["question"]
        
        # 1. RAG agent retrieves context internally and generates response
        response: AgentResponse = agent.respond(input_text)
        time.sleep(1.0)  # Gentle delay to respect API rate limits
        
        # 2. Evaluate response against reference criteria
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
            "retrieved_evidence": " ; ".join(response.retrieved_evidence),
            "latency_seconds": response.latency_seconds,
            "model": response.model,
            "error": response.error or "",
        }
        records.append(record)
        print(f"[{idx+1}/{len(df_questions)}] {qid} -> Decision: {response.decision} | Evidence: {response.retrieved_evidence[:2]} | Correct: {eval_metrics['overall_correct']}")

    df_results = pd.DataFrame(records)
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(target_out, index=False, encoding="utf-8")
    print(f"Successfully saved {len(df_results)} RAG evaluation rows to: {target_out}")
    
    return df_results

def print_stage_comparison(stage1_path: Path, stage2_path: Path) -> None:
    """Print comparative table of Stage 1 vs Stage 2 only when both genuine results exist."""
    if not stage1_path.exists():
        print("\n[NOTE] Stage 1 genuine evaluation results not found. Stage 1 vs Stage 2 comparison is PENDING.")
        return
    if not stage2_path.exists():
        print("\n[NOTE] Stage 2 genuine evaluation results not found. Stage 1 vs Stage 2 comparison is PENDING.")
        return
        
    df1 = pd.read_csv(stage1_path, dtype=str)
    df2 = pd.read_csv(stage2_path, dtype=str)
    
    print("\n" + "="*60)
    print("STAGE 1 (PROMPT BASELINE) vs STAGE 2 (GROUNDED RAG) COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} | {'Stage 1 (Baseline)':<15} | {'Stage 2 (RAG)':<15}")
    print("-" * 66)
    
    def get_acc(df, col):
        m = (df[col] == True) | (df[col] == "True")
        return f"{m.mean()*100:.1f}% ({m.sum()}/{len(df)})"
        
    print(f"{'Decision Accuracy':<30} | {get_acc(df1, 'decision_match'):<15} | {get_acc(df2, 'decision_match'):<15}")
    print(f"{'Must-Include Fact Recall':<30} | {get_acc(df1, 'must_include_passed'):<15} | {get_acc(df2, 'must_include_passed'):<15}")
    print(f"{'Must-Not-Include Passed':<30} | {get_acc(df1, 'must_not_include_passed'):<15} | {get_acc(df2, 'must_not_include_passed'):<15}")
    print(f"{'Overall Grounded Correctness':<30} | {get_acc(df1, 'overall_correct'):<15} | {get_acc(df2, 'overall_correct'):<15}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 Grounded RAG Answer Evaluation.")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all questions.")
    parser.add_argument("--asr", action="store_true", help="Use asr_text instead of clean question text.")
    args = parser.parse_args()
    
    try:
        agent = RAGAgent()
        print(f"Initialized RAGAgent with model: {agent.model}")
        results = run_rag_answer_evaluation(agent, resume=not args.force, use_asr_text=args.asr)
        print_evaluation_summary(results)
        print_stage_comparison(RESULTS_DIR / "stage1_baseline.csv", STAGE2_OUTPUT_CSV)
    except ValueError as e:
        print(f"\n[BLOCKED] Authentication Error: {e}")
        print("Configure GEMINI_API_KEY in your local environment or .env file.")
        print("Stage 2 answer evaluation remains PENDING until credentials are provided.")