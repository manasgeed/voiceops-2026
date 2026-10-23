"""Multi-Turn Evaluation Runner for Stage 3: Conversational Support Agent.

Processes practice/conversations.csv (23 conversations, 96 turns) in conversation_id
and turn order through ConversationalAgent. Evaluates decisions, tool calls,
must_include facts, and must_not_include constraints.
Saves results to results/stage3_conversation_evaluation.csv and results/stage3_conversation_metrics.json.
"""
import os
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.config import PRACTICE_DIR, RESULTS_DIR
from src.utils.metrics import evaluate_decision, evaluate_must_include, evaluate_must_not_include
from src.agent.conversational_agent import ConversationalAgent
from src.baseline.agent import AgentResponse

STAGE3_OUTPUT_CSV = RESULTS_DIR / "stage3_conversation_evaluation.csv"
STAGE3_OUTPUT_JSON = RESULTS_DIR / "stage3_conversation_metrics.json"


def load_practice_conversations(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate practice/conversations.csv."""
    path = csv_path or (PRACTICE_DIR / "conversations.csv")
    if not path.exists():
        raise FileNotFoundError(f"Conversations benchmark not found at: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(df) != 96:
        print(f"Warning: Expected 96 turns in benchmark, found {len(df)}")
    return df


def evaluate_tool_calls(actual_tool_calls: List[Dict[str, Any]], expected_tool_calls_spec: str) -> bool:
    """Evaluate whether actual executed tool calls match expected tool calls."""
    if not expected_tool_calls_spec or str(expected_tool_calls_spec).strip() in ("", "nan"):
        return len(actual_tool_calls) == 0
        
    expected_spec = expected_tool_calls_spec.strip()
    if expected_spec == "create_ticket()":
        return any(tc.get("tool") == "create_ticket" for tc in actual_tool_calls)
        
    actual_call_strs = [tc.get("call_str", "") for tc in actual_tool_calls if tc.get("call_str")]
    return any(cs == expected_spec for cs in actual_call_strs)


def run_conversations_evaluation(
    agent: ConversationalAgent,
    csv_path: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    output_json: Optional[Path] = None,
    resume: bool = True,
    max_conversations: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute multi-turn conversation evaluation across all conversations."""
    df_convs = load_practice_conversations(csv_path)
    target_csv = output_csv or STAGE3_OUTPUT_CSV
    target_json = output_json or STAGE3_OUTPUT_JSON
    
    unique_cids = list(df_convs["conversation_id"].unique())
    if max_conversations:
        unique_cids = unique_cids[:max_conversations]
        df_convs = df_convs[df_convs["conversation_id"].isin(unique_cids)]

    existing_results: Dict[str, Dict[str, Any]] = {}
    if resume and target_csv.exists():
        try:
            df_existing = pd.read_csv(target_csv, dtype=str, keep_default_na=False)
            for _, row in df_existing.iterrows():
                key = f"{row['conversation_id']}_{row['turn']}"
                if not row.get("error"):
                    existing_results[key] = row.to_dict()
            print(f"Resuming multi-turn evaluation: found {len(existing_results)} already completed turn records.")
        except Exception as e:
            print(f"Could not read existing results, starting fresh: {e}")

    records: List[Dict[str, Any]] = []
    
    for cid in unique_cids:
        sub_df = df_convs[df_convs["conversation_id"] == cid].sort_values("turn")
        scenario = sub_df.iloc[0].get("scenario", "")
        language = sub_df.iloc[0].get("language", "en")
        
        # Reset session for isolation
        agent.manager.reset_session(cid)
        print(f"\n--- Conversation {cid} (Scenario: {scenario}, Language: {language}, Turns: {len(sub_df)}) ---")
        
        for _, row in sub_df.iterrows():
            turn_idx = str(row["turn"])
            key = f"{cid}_{turn_idx}"
            
            if resume and key in existing_results:
                records.append(existing_results[key])
                print(f"  [Turn {turn_idx}] Cached from previous run.")
                continue
                
            cust_msg = row["customer_message"]
            
            # Execute turn through conversational agent
            response: AgentResponse = agent.respond(
                message=cust_msg,
                conversation_id=cid,
                language=language,
                scenario=scenario,
            )
            time.sleep(1.0)  # Gentle delay to respect API rate limits
            
            # Evaluate decision
            exp_decision = row.get("expected_decision", "")
            dec_match = evaluate_decision(response.decision, exp_decision, "")
            
            # Evaluate tool calls
            exp_tools = row.get("expected_tool_calls", "")
            tools_match = evaluate_tool_calls(response.tool_calls, exp_tools)
            
            # Evaluate must_include
            must_inc = row.get("must_include", "")
            inc_passed, inc_score, missing = evaluate_must_include(response.reply_text, must_inc)
            
            # Evaluate must_not_include
            must_not = row.get("must_not_include", "")
            not_inc_passed, violations = evaluate_must_not_include(response.reply_text, must_not)
            
            # Overall correctness
            overall_correct = dec_match and tools_match and inc_passed and not_inc_passed
            
            tool_calls_str = " ; ".join(tc.get("call_str", tc.get("tool", "")) for tc in response.tool_calls)
            
            record = {
                "conversation_id": cid,
                "turn": turn_idx,
                "scenario": scenario,
                "language": language,
                "order_id": row.get("order_id", ""),
                "customer_message": cust_msg,
                "agent_reply": response.reply_text,
                "predicted_decision": response.decision,
                "expected_decision": exp_decision,
                "decision_match": dec_match,
                "predicted_tool_calls": tool_calls_str,
                "expected_tool_calls": exp_tools,
                "tools_match": tools_match,
                "must_include": must_inc,
                "must_include_passed": inc_passed,
                "must_include_score": inc_score,
                "missing_facts": " ; ".join(missing),
                "must_not_include": must_not,
                "must_not_include_passed": not_inc_passed,
                "violations": " ; ".join(violations),
                "overall_correct": overall_correct,
                "retrieved_evidence": " ; ".join(response.retrieved_evidence),
                "latency_seconds": response.latency_seconds,
                "model": response.model,
                "error": response.error or "",
            }
            records.append(record)
            print(f"  [Turn {turn_idx}] Dec: {response.decision} (exp: {exp_decision}) | Tools: {tool_calls_str or 'none'} | Correct: {overall_correct}")

    # Save to CSV
    df_results = pd.DataFrame(records)
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(target_csv, index=False, encoding="utf-8")
    print(f"\nSaved {len(df_results)} multi-turn evaluation records to: {target_csv}")
    
    # Compute aggregate metrics
    metrics = compute_conversation_metrics(df_results)
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved aggregate conversation metrics to: {target_json}")
    
    return df_results, metrics


def compute_conversation_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate conversation metrics."""
    total_turns = len(df)
    if total_turns == 0:
        return {}
        
    dec_matches = (df["decision_match"] == True) | (df["decision_match"] == "True")
    tools_matches = (df["tools_match"] == True) | (df["tools_match"] == "True")
    inc_passed = (df["must_include_passed"] == True) | (df["must_include_passed"] == "True")
    not_inc_passed = (df["must_not_include_passed"] == True) | (df["must_not_include_passed"] == "True")
    overall_correct = (df["overall_correct"] == True) | (df["overall_correct"] == "True")
    
    # Conversation-level completion (all turns in conversation correct)
    conv_groups = df.groupby("conversation_id")["overall_correct"].apply(
        lambda s: all(v is True or v == "True" for v in s)
    )
    perfect_conversations = int(conv_groups.sum())
    total_conversations = int(conv_groups.count())
    
    return {
        "total_conversations": total_conversations,
        "perfect_conversations": perfect_conversations,
        "conversation_success_rate": round(perfect_conversations / total_conversations, 4),
        "total_turns": total_turns,
        "decision_accuracy": round(float(dec_matches.mean()), 4),
        "tool_call_accuracy": round(float(tools_matches.mean()), 4),
        "must_include_pass_rate": round(float(inc_passed.mean()), 4),
        "must_not_include_pass_rate": round(float(not_inc_passed.mean()), 4),
        "turn_overall_correctness": round(float(overall_correct.mean()), 4),
    }


def print_conversation_summary(metrics: Dict[str, Any]) -> None:
    """Print quantitative summary of Stage 3 conversation evaluation."""
    print("\n" + "="*60)
    print("STAGE 3 CONVERSATIONAL AGENT: EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Conversations:         {metrics.get('total_conversations')}")
    print(f"Perfect Conversations:       {metrics.get('perfect_conversations')}/{metrics.get('total_conversations')} ({metrics.get('conversation_success_rate', 0)*100:.1f}%)")
    print(f"Total Turns Evaluated:       {metrics.get('total_turns')}")
    print(f"Decision Accuracy:           {metrics.get('decision_accuracy', 0)*100:.1f}%")
    print(f"Tool Call Accuracy:          {metrics.get('tool_call_accuracy', 0)*100:.1f}%")
    print(f"Must-Include Fact Recall:    {metrics.get('must_include_pass_rate', 0)*100:.1f}%")
    print(f"Must-Not-Include Passed:     {metrics.get('must_not_include_pass_rate', 0)*100:.1f}%")
    print(f"Turn Overall Correctness:    {metrics.get('turn_overall_correctness', 0)*100:.1f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 3 Conversational Evaluation.")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all conversations.")
    parser.add_argument("--max-convs", type=int, default=None, help="Limit number of conversations to evaluate.")
    args = parser.parse_args()
    
    agent = ConversationalAgent()
    print(f"Initialized ConversationalAgent with model: {agent.model}")
    df, metrics = run_conversations_evaluation(agent, resume=not args.force, max_conversations=args.max_convs)
    print_conversation_summary(metrics)
