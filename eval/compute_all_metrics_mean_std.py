"""
Compute cross-lingual consistency metrics with Transfer/Non-Transfer
and per-cultural-label breakdown.

All metrics are computed per template (5 templates), then reported as
mean ± std across templates for each model.

Metrics:
  - Accuracy (FA, EN)
  - Binary consistency (Both/Overall, EN-only, FA-only, Neither, Consist)
  - Transfer: P(correct in FA | correct in EN)
  - Non-Transfer: P(correct in FA | incorrect in EN)
  - RankC (Spearman rank correlation of candidate scores across languages)

Usage:
    python eval/compute_all_metrics_mean_std.py --data_dir data/results/random/all_templates --fa_prefix evaluate --en_prefix evaluate
"""

import json
import os
import argparse
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

NUM_TEMPLATES = 5


# ═══════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════

def load_model_data(data_dir, model_name, fa_prefix="evaluate", en_prefix="evaluate"):
    """Load FA and EN evaluation files for a given model."""
    fa_path = os.path.join(data_dir, f"{fa_prefix}_{model_name}_all_relations_fa.json")
    en_path = os.path.join(data_dir, f"{en_prefix}_{model_name}_all_relations_en.json")
    with open(fa_path) as f:
        fa_data = json.load(f)
    with open(en_path) as f:
        en_data = json.load(f)
    return fa_data, en_data


def get_candidate_keys(item, lang):
    """
    Get all candidate keys in canonical order: [correct, distractor1, ..., distractor4].
    """
    if lang == "en":
        return [item["objects_en"][0]] + list(item["object_candidates_en"])
    else:
        return [item["objects"][0]] + list(item["object_candidates"])


def get_correct_key(item, lang):
    """Get the correct answer as it appears in scores."""
    if lang == "en":
        return item["objects_en"][0]
    else:
        return item["objects"][0]


def get_score_vector_for_template(item, lang, t):
    """
    Get the score vector for template t in canonical candidate order.
    Returns list of floats, or None if any candidate is missing.
    """
    keys = get_candidate_keys(item, lang)
    template_scores = item["per_template_scores"][t]
    scores = []
    for k in keys:
        if k not in template_scores:
            return None
        scores.append(template_scores[k])
    return scores


def get_rank_of_correct_for_template(item, lang, t):
    """
    Get the rank (1-indexed) of the correct answer for template t.
    Rank 1 = highest score.
    """
    correct = get_correct_key(item, lang)
    template_scores = item["per_template_scores"][t]
    sorted_candidates = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (candidate, score) in enumerate(sorted_candidates, start=1):
        if candidate == correct:
            return rank
    return len(template_scores)


# ═══════════════════════════════════════════════════════════════════════════
# Per-template metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics_for_template(fa_items, en_items, t):
    """
    Compute all metrics using template t's scores and correctness.
    """
    N = len(fa_items)
    if N == 0:
        return None
    assert len(en_items) == N

    # Binary correctness for template t
    fa_correct = np.array([item["per_template_correct"][t] for item in fa_items], dtype=bool)
    en_correct = np.array([item["per_template_correct"][t] for item in en_items], dtype=bool)

    # ── Accuracy ──
    acc_fa = fa_correct.mean()
    acc_en = en_correct.mean()

    # ── Binary Consistency ──
    both = (fa_correct & en_correct).mean()
    en_only = (~fa_correct & en_correct).mean()
    fa_only = (fa_correct & ~en_correct).mean()
    neither = (~fa_correct & ~en_correct).mean()
    consist = (fa_correct == en_correct).mean()

    # ── Transfer & Non-Transfer ──
    n_en_correct = en_correct.sum()
    n_en_incorrect = (~en_correct).sum()

    if n_en_correct > 0:
        transfer = (fa_correct & en_correct).sum() / n_en_correct
    else:
        transfer = float("nan")

    if n_en_incorrect > 0:
        non_transfer = (fa_correct & ~en_correct).sum() / n_en_incorrect
    else:
        non_transfer = float("nan")

    # ── RankC ──
    per_item_corr = []
    for i in range(N):
        fa_scores = get_score_vector_for_template(fa_items[i], "fa", t)
        en_scores = get_score_vector_for_template(en_items[i], "en", t)
        if fa_scores is None or en_scores is None:
            continue
        if len(fa_scores) != len(en_scores) or len(fa_scores) < 3:
            continue
        corr, _ = spearmanr(fa_scores, en_scores)
        if not np.isnan(corr):
            per_item_corr.append(corr)

    rankc = np.mean(per_item_corr) if per_item_corr else float("nan")

    return {
        "n": N,
        "acc_fa": acc_fa * 100,
        "acc_en": acc_en * 100,
        "both": both * 100,
        "en_only": en_only * 100,
        "fa_only": fa_only * 100,
        "neither": neither * 100,
        "consist": consist * 100,
        "transfer": transfer * 100,
        "non_transfer": non_transfer * 100,
        "rankc": rankc * 100,
    }


METRIC_KEYS = [
    "acc_en", "acc_fa", "transfer", "non_transfer",
    "both", "en_only", "fa_only", "neither", "consist",
    "rankc",
]


def compute_mean_std_across_templates(fa_items, en_items):
    """
    Compute metrics for each of the 5 templates, then return
    mean and std for every metric.
    """
    per_template = []
    for t in range(NUM_TEMPLATES):
        m = compute_metrics_for_template(fa_items, en_items, t)
        if m is not None:
            per_template.append(m)

    if not per_template:
        return None

    result = {"n": per_template[0]["n"]}
    for key in METRIC_KEYS:
        vals = [m[key] for m in per_template if not np.isnan(m[key])]
        if vals:
            result[f"{key}_mean"] = np.mean(vals)
            result[f"{key}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        else:
            result[f"{key}_mean"] = float("nan")
            result[f"{key}_std"] = float("nan")

    return result


def filter_by_label(fa_data, en_data, label):
    """Filter FA and EN items by cultural label. Label=None returns all."""
    if label is None:
        return fa_data, en_data
    fa_filtered, en_filtered = [], []
    for i in range(len(fa_data)):
        item_label = en_data[i].get("label", fa_data[i].get("label", None))
        if item_label == label:
            fa_filtered.append(fa_data[i])
            en_filtered.append(en_data[i])
    return fa_filtered, en_filtered


# ═══════════════════════════════════════════════════════════════════════════
# Pretty printing
# ═══════════════════════════════════════════════════════════════════════════

DISPLAY_METRICS = [
    ("acc_en",       "Acc_EN"),
    ("acc_fa",       "Acc_FA"),
    ("transfer",     "Transf"),
    ("non_transfer", "NonTr"),
    ("both",         "Overall"),
    ("consist",      "Consist"),
    ("rankc",        "RankC"),
]

COL_WIDTH = 12  # width per metric column


def fmt_mean_std(mean_val, std_val, precision=1):
    """Format as 'mean±std' string."""
    if np.isnan(mean_val):
        return "n/a"
    return f"{mean_val:.{precision}f}±{std_val:.{precision}f}"


def print_metrics_table(all_results, label_name, models):
    """Print a formatted table with mean±std across templates."""
    print(f"\n{'='*170}")
    print(f"  {label_name}")
    print(f"{'='*170}")

    # Header
    parts = [f"{'Model':<42}", f"{'N':>5}"]
    for _, col_name in DISPLAY_METRICS:
        parts.append(f"{col_name:>{COL_WIDTH}}")
    header = " ".join(parts)
    print(header)
    print("-" * len(header))

    for model in models:
        m = all_results[model]
        if m is None:
            print(f"{model:<42} {'(no data)':>5}")
            continue

        parts = [f"{model:<42}", f"{m['n']:>5}"]
        for key, _ in DISPLAY_METRICS:
            cell = fmt_mean_std(m[f"{key}_mean"], m[f"{key}_std"])
            parts.append(f"{cell:>{COL_WIDTH}}")
        print(" ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-lingual metrics (mean±std across templates) with per-label breakdown"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing evaluation JSON files")
    parser.add_argument("--fa_prefix", type=str, default="hard_evaluate",
                        help="Prefix for FA files (default: hard_evaluate)")
    parser.add_argument("--en_prefix", type=str, default="evaluate",
                        help="Prefix for EN files (default: evaluate)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional CSV output path")
    args = parser.parse_args()

    # Auto-discover models from EN filenames
    models = set()
    for fname in os.listdir(args.data_dir):
        if fname.startswith(f"{args.en_prefix}_") and fname.endswith("_en.json"):
            model_name = fname.replace(f"{args.en_prefix}_", "").replace("_all_relations_en.json", "")
            fa_fname = f"{args.fa_prefix}_{model_name}_all_relations_fa.json"
            if os.path.exists(os.path.join(args.data_dir, fa_fname)):
                models.add(model_name)

    models = sorted(models)
    print(f"Found {len(models)} models: {models}")

    labels = [
        (None, "All items"),
        ("culturally-neutral", "Culturally neutral (culture-general)"),
        ("persian-centric", "Persian-centric (culture-specific)")
    ]

    all_label_results = {}

    for label, label_name in labels:
        label_results = {}
        for model in models:
            fa_data, en_data = load_model_data(
                args.data_dir, model,
                fa_prefix=args.fa_prefix,
                en_prefix=args.en_prefix,
            )
            fa_subset, en_subset = filter_by_label(fa_data, en_data, label)
            label_results[model] = compute_mean_std_across_templates(fa_subset, en_subset)
        all_label_results[label_name] = label_results
        print_metrics_table(label_results, label_name, models)

    # ── Optional CSV output ──
    if args.output:
        import csv

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            header = ["label", "model", "n"]
            for key in METRIC_KEYS:
                header.append(f"{key}_mean")
                header.append(f"{key}_std")
            writer.writerow(header)

            for label_name, label_results in all_label_results.items():
                for model in models:
                    m = label_results[model]
                    if m is None:
                        continue
                    row = [label_name, model, m["n"]]
                    for key in METRIC_KEYS:
                        row.append(f"{m[f'{key}_mean']:.4f}")
                        row.append(f"{m[f'{key}_std']:.4f}")
                    writer.writerow(row)

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()