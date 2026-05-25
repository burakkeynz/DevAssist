# Importing required libraries for benchmark visualization and plotting
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any

# Configuring logging for visualization pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Defining paths for results and plots
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")
QUESTIONS_PATH = Path("eval/questions.json")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Defining project and category color mappings
PROJECT_COLORS = {
    "agentbench_tr": "#8b5cf6",
    "aeroguard":     "#3b82f6",
    "highway_env":   "#10b981",
    "general":       "#f59e0b"
}

CATEGORY_COLORS = {
    "retrieval":  "#8b5cf6",
    "agent":      "#3b82f6",
    "modeling":   "#10b981",
    "features":   "#f59e0b",
    "evaluation": "#ef4444",
    "alert":      "#06b6d4",
    "rl":         "#ec4899",
    "xai":        "#a78bfa",
    "memory":     "#34d399"
}

# Applying dark theme for all matplotlib figures
plt.rcParams.update({
    "figure.facecolor":  "#0e0e0e",
    "axes.facecolor":    "#141414",
    "axes.edgecolor":    "#2a2a2a",
    "axes.labelcolor":   "#a0a0a0",
    "axes.titlecolor":   "#f2f2f2",
    "xtick.color":       "#606060",
    "ytick.color":       "#606060",
    "text.color":        "#f2f2f2",
    "grid.color":        "#1e1e1e",
    "grid.linewidth":    0.8,
    "font.family":       "monospace",
    "figure.dpi":        150
})


# Loading benchmark results from most recent JSON file
def load_benchmark_results() -> List[Dict[str, Any]]:
    result_files = sorted(RESULTS_DIR.glob("benchmark_base_rag_*.json"), reverse=True)
    if not result_files:
        raise FileNotFoundError("No benchmark_base_rag_*.json found in results/")
    latest = result_files[0]
    logger.info(f"Loading benchmark results from: {latest}")
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


# Loading questions with expected keywords from JSON
def load_questions() -> Dict[str, Dict[str, Any]]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["questions"]}


# Computing keyword hit rate for each response against expected keywords
def compute_keyword_hits(
    results: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    hits = []
    for r in results:
        qid = r["id"]
        q = questions.get(qid, {})
        keywords = q.get("expected_keywords", [])
        response = r.get("response", "").lower()

        if not keywords:
            hit_rate = 0.0
            matched = []
        else:
            matched = [kw for kw in keywords if kw.lower() in response]
            hit_rate = round(len(matched) / len(keywords), 4)

        hits.append({
            "id": qid,
            "question": r["question"][:40],
            "category": r["category"],
            "project": r["project"],
            "hit_rate": hit_rate,
            "matched": matched,
            "total_keywords": len(keywords)
        })
    logger.info(f"Computing keyword hit rates for {len(hits)} results...")
    return hits


# Generating attribution heatmap across all questions and top chunks
def plot_attribution_heatmap(results: List[Dict[str, Any]]) -> None:
    logger.info("Generating TreeRAG attribution heatmap...")

    ids = [r["id"] for r in results]
    top_attributions = []

    for r in results:
        attr = r.get("top_attribution_pct", 0)
        top_attributions.append(attr)

    # Building 20×1 heatmap data — top attribution per question
    data = np.array(top_attributions).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(16, 3))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#141414")

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="Purples",
        vmin=0,
        vmax=100,
        interpolation="nearest"
    )

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title(
        "TreeRAG Attribution Heatmap — Top Chunk Attribution % per Query",
        fontsize=11,
        pad=12,
        color="#f2f2f2"
    )

    # Adding attribution percentage annotations...
    for j, val in enumerate(top_attributions):
        color = "#f2f2f2" if val < 60 else "#0e0e0e"
        ax.text(j, 0, f"{val:.1f}%", ha="center", va="center",
                fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.08, pad=0.35, shrink=0.6)
    cbar.ax.tick_params(colors="#606060", labelsize=8)
    cbar.set_label("Attribution %", color="#a0a0a0", fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "attribution_heatmap.png"
    plt.savefig(out, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close()
    logger.info(f"Saving attribution heatmap to: {out}")


# Generating latency bar chart with category color coding
def plot_latency_chart(results: List[Dict[str, Any]]) -> None:
    logger.info("Generating latency distribution chart...")

    ids = [r["id"] for r in results]
    latencies = [r["latency_sec"] for r in results]
    categories = [r["category"] for r in results]
    colors = [CATEGORY_COLORS.get(c, "#606060") for c in categories]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#141414")

    bars = ax.bar(ids, latencies, color=colors, alpha=0.85, width=0.65, zorder=3)

    # Adding value labels on bars
    for bar, val in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}s",
            ha="center", va="bottom",
            fontsize=7.5, color="#a0a0a0"
        )

    avg = np.mean(latencies)
    ax.axhline(avg, color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.7, zorder=4)
    ax.text(len(ids) - 0.5, avg + 0.4, f"avg {avg:.1f}s",
            color="#f59e0b", fontsize=8, ha="right")

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Latency (seconds)", fontsize=9)
    ax.set_title("Query Latency Distribution — DevAssist TreeRAG Pipeline",
                 fontsize=11, pad=12)
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Adding category legend
    unique_cats = list(dict.fromkeys(categories))
    patches = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, "#606060"), label=c)
        for c in unique_cats
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.15,
              facecolor="#1a1a1a", edgecolor="#2a2a2a", loc="upper right")

    plt.tight_layout()
    out = PLOTS_DIR / "latency_chart.png"
    plt.savefig(out, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close()
    logger.info(f"Saving latency chart to: {out}")


# Generating retrieval metrics summary bar chart
def plot_retrieval_metrics(results: List[Dict[str, Any]]) -> None:
    logger.info("Generating retrieval metrics summary chart...")

    total = len(results)
    rag_count = sum(1 for r in results if r.get("rag_mode") == "rag")
    hit_k = sum(
        1 for r in results
        if r.get("attribution_count", 0) > 0 and r.get("top_attribution_pct", 0) > 10
    ) / total

    # Computing MRR
    rr_list = []
    for r in results:
        attr = r.get("top_attribution_pct", 0)
        if attr > 50:
            rr_list.append(1.0)
        elif attr > 20:
            rr_list.append(0.5)
        elif attr > 10:
            rr_list.append(0.33)
        else:
            rr_list.append(0.0)
    mrr = np.mean(rr_list)
    rag_rate = rag_count / total

    metrics = {
        "Hit@5": hit_k,
        "MRR": mrr,
        "RAG Activation": rag_rate
    }
    metric_colors = ["#8b5cf6", "#3b82f6", "#10b981"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#141414")

    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values()),
        color=metric_colors,
        alpha=0.85,
        width=0.45,
        zorder=3
    )

    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=13, color="#f2f2f2", fontweight="bold"
        )

    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Retrieval Evaluation Metrics — base_rag Baseline",
                 fontsize=11, pad=12)
    ax.axhline(1.0, color="#2a2a2a", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "retrieval_metrics.png"
    plt.savefig(out, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close()
    logger.info(f"Saving retrieval metrics chart to: {out}")


# Generating category-level average attribution bar chart
def plot_category_attribution(results: List[Dict[str, Any]]) -> None:
    logger.info("Generating category attribution comparison chart...")

    cat_data: Dict[str, List[float]] = {}
    for r in results:
        cat = r["category"]
        attr = r.get("top_attribution_pct", 0)
        cat_data.setdefault(cat, []).append(attr)

    categories = list(cat_data.keys())
    averages = [np.mean(cat_data[c]) for c in categories]
    colors = [CATEGORY_COLORS.get(c, "#606060") for c in categories]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#141414")

    bars = ax.bar(categories, averages, color=colors, alpha=0.85,
                  width=0.55, zorder=3)

    for bar, val in zip(bars, averages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=9, color="#f2f2f2", fontweight="bold"
        )

    ax.set_ylabel("Avg Top Attribution %", fontsize=9)
    ax.set_title("TreeRAG Attribution by Query Category",
                 fontsize=11, pad=12)
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "category_attribution.png"
    plt.savefig(out, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close()
    logger.info(f"Saving category attribution chart to: {out}")


# Generating keyword hit rate precision bar chart per question
def plot_keyword_hit_rate(
    results: List[Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]]
) -> None:
    logger.info("Generating keyword hit rate precision chart...")

    hits = compute_keyword_hits(results, questions)
    ids = [h["id"] for h in hits]
    rates = [h["hit_rate"] * 100 for h in hits]
    projects = [h["project"] for h in hits]
    colors = [PROJECT_COLORS.get(p, "#606060") for p in projects]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#141414")

    bars = ax.bar(ids, rates, color=colors, alpha=0.85, width=0.65, zorder=3)

    for bar, val in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.0f}%",
            ha="center", va="bottom",
            fontsize=7.5, color="#a0a0a0"
        )

    avg = np.mean(rates)
    ax.axhline(avg, color="#f59e0b", linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=4)
    ax.text(len(ids) - 0.5, avg + 2,
            f"avg {avg:.1f}%", color="#f59e0b", fontsize=8, ha="right")

    ax.set_ylim(0, 115)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Keyword Hit Rate %", fontsize=9)
    ax.set_title(
        "Ground Truth Keyword Precision — Expected Keywords in Response",
        fontsize=11, pad=12
    )
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Adding project legend
    patches = [
        mpatches.Patch(color=v, label=k)
        for k, v in PROJECT_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.15,
              facecolor="#1a1a1a", edgecolor="#2a2a2a", loc="upper right")

    plt.tight_layout()
    out = PLOTS_DIR / "keyword_hit_rate.png"
    plt.savefig(out, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close()
    logger.info(f"Saving keyword hit rate chart to: {out}")


# Running full visualization pipeline and generating all plots
def main() -> None:
    logger.info("Starting DevAssist benchmark visualization pipeline...")

    results = load_benchmark_results()
    questions = load_questions()

    plot_attribution_heatmap(results)
    plot_latency_chart(results)
    plot_retrieval_metrics(results)
    plot_category_attribution(results)
    plot_keyword_hit_rate(results, questions)

    logger.info(
        f"Completing visualization pipeline — "
        f"5 plots saved to: {PLOTS_DIR}"
    )

    print("\n--- Visualization Complete ---")
    print(f"Plots saved to: {PLOTS_DIR}")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  ✓ {p.name}")


if __name__ == "__main__":
    main()