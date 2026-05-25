# Importing required libraries for benchmark metrics computation
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from rouge_score import rouge_scorer
import logging

# Configuring logging for metrics computation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# Loading benchmark results from JSON file
def load_results(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Computing ROUGE scores for response quality evaluation
def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []

    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            continue
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": float(round(np.mean(r1_scores) if r1_scores else 0, 4)),
        "rouge2": float(round(np.mean(r2_scores) if r2_scores else 0, 4)),
        "rougeL": float(round(np.mean(rl_scores) if rl_scores else 0, 4))
    }


# Computing Hit@K metric — whether relevant chunk retrieved in top-K...
def compute_hit_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    hits = 0
    for r in results:
        if r.get("attribution_count", 0) > 0 and r.get("top_attribution_pct", 0) > 10:
            hits += 1
    return round(hits / len(results), 4) if results else 0.0


# Computing Mean Reciprocal Rank for retrieval evaluation...
def compute_mrr(results: List[Dict[str, Any]]) -> float:
    reciprocal_ranks = []
    for r in results:
        attribution = r.get("top_attribution_pct", 0)
        if attribution > 50:
            reciprocal_ranks.append(1.0)
        elif attribution > 20:
            reciprocal_ranks.append(0.5)
        elif attribution > 10:
            reciprocal_ranks.append(0.33)
        else:
            reciprocal_ranks.append(0.0)
    return float(round(np.mean(reciprocal_ranks) if reciprocal_ranks else 0, 4))


# Building comprehensive benchmark CSV from results
def build_benchmark_csv(results: List[Dict[str, Any]], output_path: str) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "id": r["id"],
            "question": r["question"][:60],
            "category": r["category"],
            "project": r["project"],
            "baseline": r["baseline"],
            "latency_sec": r["latency_sec"],
            "rag_mode": r["rag_mode"],
            "top_attribution_pct": r["top_attribution_pct"],
            "top_chunk": r["top_chunk"],
            "response_length_words": r["response_length_words"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saving benchmark CSV to: {output_path}")
    return df


# Computing and printing full metrics summary
def compute_metrics(results_path: str) -> Dict[str, Any]:
    results = load_results(results_path)
    logger.info(f"Computing metrics for {len(results)} results...")

    hit_k = compute_hit_at_k(results)
    mrr = compute_mrr(results)

    avg_latency = round(np.mean([r["latency_sec"] for r in results]), 2)
    avg_attribution = round(np.mean([r["top_attribution_pct"] for r in results]), 2)
    rag_rate = round(sum(1 for r in results if r["rag_mode"] == "rag") / len(results), 4)

    metrics = {
        "total_questions": len(results),
        "hit_at_5": hit_k,
        "mrr": mrr,
        "avg_latency_sec": avg_latency,
        "avg_top_attribution_pct": avg_attribution,
        "rag_activation_rate": rag_rate
    }

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k:30s}: {v}")

    csv_path = str(Path(results_path).parent / "benchmark_table.csv")
    build_benchmark_csv(results, csv_path)

    return metrics


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        compute_metrics(sys.argv[1])
    else:
        # Finding most recent benchmark result
        result_files = sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True)
        if result_files:
            compute_metrics(str(result_files[0]))
        else:
            logger.error("No benchmark results found in results/ directory.")