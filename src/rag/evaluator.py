"""Retrieval Evaluation Module (Phase 6).

Evaluates HybridRetriever against practice/questions.csv gold_evidence fields.
Runs completely offline without requiring LLM API credentials.
Computes Hit Rate@k, Recall@k, Precision@k, and Mean Reciprocal Rank (MRR).
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import PRACTICE_DIR, RESULTS_DIR
from src.rag.retriever import HybridRetriever, RetrievalResult

RETRIEVAL_RESULTS_CSV = RESULTS_DIR / "stage2_retrieval_evaluation.csv"
RETRIEVAL_METRICS_JSON = RESULTS_DIR / "stage2_retrieval_metrics.json"

def parse_gold_evidence(gold_str: str) -> List[str]:
    """Parse semicolon-separated gold evidence string into a list of normalized section IDs."""
    if not gold_str or str(gold_str).strip() in ("", "nan"):
        return []
    items = []
    for item in gold_str.split(";"):
        cleaned = item.strip()
        if cleaned:
            items.append(cleaned)
    return items

def evaluate_retrieval(
    retriever: Optional[HybridRetriever] = None,
    csv_path: Optional[Path] = None,
    top_k: int = 5,
    output_csv: Optional[Path] = None,
    output_json: Optional[Path] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate retrieval performance across all benchmark questions."""
    path = csv_path or (PRACTICE_DIR / "questions.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    
    if retriever is None:
        retriever = HybridRetriever()
        
    records = []
    
    hits_at_k = 0
    recalls_at_k = []
    precisions_at_k = []
    reciprocal_ranks = []
    answerable_count = 0
    
    for idx, row in df.iterrows():
        qid = row["id"]
        question = row["question"]
        gold_items = parse_gold_evidence(row.get("gold_evidence", ""))
        hard_negs = parse_gold_evidence(row.get("hard_negatives", ""))
        is_answerable = row.get("answerable", "True") == "True"
        
        # Retrieve top_k chunks
        results: List[RetrievalResult] = retriever.retrieve(question, top_k=top_k)
        retrieved_sec_ids = [r.section_id for r in results]
        
        gold_set = set(gold_items)
        retrieved_set = set(retrieved_sec_ids)
        
        if is_answerable and gold_set:
            answerable_count += 1
            intersection = gold_set.intersection(retrieved_set)
            hit = len(intersection) > 0
            recall = len(intersection) / len(gold_set)
            precision = len(intersection) / len(retrieved_sec_ids) if retrieved_sec_ids else 0.0
            
            # Reciprocal rank (first rank where a gold evidence appears)
            rr = 0.0
            for rank, sec_id in enumerate(retrieved_sec_ids, start=1):
                if sec_id in gold_set:
                    rr = 1.0 / rank
                    break
                    
            if hit:
                hits_at_k += 1
            recalls_at_k.append(recall)
            precisions_at_k.append(precision)
            reciprocal_ranks.append(rr)
        else:
            # Unanswerable question
            hit = None
            recall = None
            precision = None
            rr = None
            
        hard_neg_hits = [sec for sec in retrieved_sec_ids if sec in set(hard_negs)]
        
        records.append({
            "question_id": qid,
            "answerable": is_answerable,
            "question": question,
            "gold_evidence": " ; ".join(gold_items),
            "retrieved_section_ids": " ; ".join(retrieved_sec_ids),
            "hit_at_k": hit,
            "recall_at_k": round(recall, 3) if recall is not None else None,
            "precision_at_k": round(precision, 3) if precision is not None else None,
            "reciprocal_rank": round(rr, 3) if rr is not None else None,
            "hard_negatives": " ; ".join(hard_negs),
            "hard_negatives_retrieved": " ; ".join(hard_neg_hits),
        })

    df_out = pd.DataFrame(records)
    
    # Compute aggregate metrics
    agg_hit_rate = hits_at_k / answerable_count if answerable_count else 0.0
    agg_mrr = sum(reciprocal_ranks) / answerable_count if answerable_count else 0.0
    agg_mean_recall = sum(recalls_at_k) / answerable_count if answerable_count else 0.0
    agg_mean_precision = sum(precisions_at_k) / answerable_count if answerable_count else 0.0
    
    metrics = {
        "total_questions": len(df),
        "answerable_questions": answerable_count,
        "unanswerable_questions": len(df) - answerable_count,
        "top_k": top_k,
        "retrieval_backend": retriever.active_backend,
        "hit_rate_at_k": round(agg_hit_rate, 4),
        "hit_rate_at_k_description": "Fraction of answerable questions where at least one gold evidence section appears in top-k",
        "mean_reciprocal_rank_mrr": round(agg_mrr, 4),
        "mean_reciprocal_rank_description": "Average reciprocal rank (1/rank) of first matching gold section",
        "mean_recall_at_k": round(agg_mean_recall, 4),
        "mean_recall_at_k_description": "Average proportion of all required gold evidence sections retrieved in top-k",
        "mean_precision_at_k": round(agg_mean_precision, 4),
        "mean_precision_at_k_description": "Average proportion of retrieved top-k chunks that belong to gold evidence (bounded by len(gold)/k)",
    }
    
    # Persist outputs
    out_csv = output_csv or RETRIEVAL_RESULTS_CSV
    out_json = output_json or RETRIEVAL_METRICS_JSON
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Retrieval evaluation saved to {out_csv} and {out_json}")
    return df_out, metrics

if __name__ == "__main__":
    df_res, agg = evaluate_retrieval(top_k=5)
    print("\n" + "="*50)
    print("STAGE 2 HYBRID RETRIEVAL EVALUATION RESULTS (top_k=5)")
    print("="*50)
    for k, v in agg.items():
        print(f"{k:25s}: {v}")
    print("="*50 + "\n")
