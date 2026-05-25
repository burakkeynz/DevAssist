# Importing required libraries for benchmark evaluation pipeline
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configuring logging for benchmark operations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Defining benchmark configuration constants
API_BASE = "http://localhost:8000"
QUESTIONS_PATH = "eval/questions.json"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BASELINES = [
    "base",          # No RAG, no fine-tuning
    "base_rag",      # RAG enabled, no fine-tuning
    "finetuned_rag"  # RAG + fine-tuned model (future)
]


# Loading benchmark questions from JSON file
def load_questions() -> List[Dict[str, Any]]:
    with open(QUESTIONS_PATH, "r") as f:
        data = json.load(f)
    return data["questions"]


# Querying DevAssist API and collecting SSE streaming response
def query_api(question: str, use_rag: bool = True) -> Dict[str, Any]:
    start_time = time.time()
    full_response = ""
    attribution_data = []
    mode = "unknown"

    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={"query": question},
            stream=True,
            timeout=120
        )

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:"):
                continue
            raw = decoded[5:].strip()
            try:
                event = json.loads(raw)
                if event.get("type") == "token":
                    full_response += event.get("data", "")
                elif event.get("type") == "attribution":
                    attribution_data = event.get("data", [])
                    mode = event.get("mode", "unknown")
            except Exception:
                continue

    except Exception as e:
        logger.error(f"Failing API query: {e}")
        full_response = ""

    elapsed = round(time.time() - start_time, 2)

    return {
        "response": full_response,
        "attribution": attribution_data,
        "mode": mode,
        "latency_sec": elapsed,
        "response_length": len(full_response.split())
    }


# Running benchmark for single baseline across all questions
def run_baseline(
    baseline: str,
    questions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    logger.info(f"Running benchmark baseline: {baseline}")
    results = []

    for q in questions:
        logger.info(f"Evaluating question {q['id']}: {q['question'][:50]}...")

        result = query_api(q["question"])

        top_attribution = 0.0
        top_chunk = "N/A"
        if result["attribution"]:
            top = result["attribution"][0]
            top_attribution = top.get("attribution_pct", 0)
            top_chunk = top.get("function_name", "N/A")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "project": q["project"],
            "baseline": baseline,
            "response": result["response"],
            "response_length_words": result["response_length"],
            "latency_sec": result["latency_sec"],
            "rag_mode": result["mode"],
            "top_attribution_pct": top_attribution,
            "top_chunk": top_chunk,
            "attribution_count": len(result["attribution"])
        })

        logger.info(
            f"Completed {q['id']} — "
            f"latency: {result['latency_sec']}s, "
            f"mode: {result['mode']}, "
            f"top attribution: {top_attribution}%"
        )

        # Pausing between queries to avoid overloading local inference
        time.sleep(2)

    return results


# Saving benchmark results to JSON file
def save_results(results: List[Dict[str, Any]], baseline: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"benchmark_{baseline}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saving benchmark results to: {output_path}")


# Executing full benchmark pipeline across all baselines
def main() -> None:
    logger.info("Starting DevAssist benchmark evaluation pipeline...")
    questions = load_questions()
    logger.info(f"Loaded {len(questions)} evaluation questions...")

    all_results = []

    # Running base_rag baseline — primary evaluation
    results = run_baseline("base_rag", questions)
    save_results(results, "base_rag")
    all_results.extend(results)

    logger.info(
        f"Completing benchmark — "
        f"total questions: {len(all_results)}, "
        f"baseline: base_rag"
    )

    # Printing summary statistics
    avg_latency = sum(r["latency_sec"] for r in all_results) / len(all_results)
    avg_attribution = sum(r["top_attribution_pct"] for r in all_results) / len(all_results)
    rag_count = sum(1 for r in all_results if r["rag_mode"] == "rag")

    print("\n--- Benchmark Summary ---")
    print(f"Total questions   : {len(all_results)}")
    print(f"RAG mode activated: {rag_count}/{len(all_results)}")
    print(f"Avg latency       : {avg_latency:.1f}s")
    print(f"Avg top attribution: {avg_attribution:.1f}%")


if __name__ == "__main__":
    main()