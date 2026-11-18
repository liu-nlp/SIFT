"""
Compute cross-lingual consistency metrics with Transfer/Non-Transfer
and per-cultural-label breakdown:
  - Accuracy (FA, EN)
  - Binary consistency (Both/Overall, EN-only, FA-only, Neither, Consist)
  - Transfer: P(correct in FA | correct in EN)
  - Non-Transfer: P(correct in FA | incorrect in EN)
  - MRR (FA, EN, MRR∧)
  - RankC (Spearman rank correlation of candidate scores across languages)

All metrics are computed:
  1. Over all items
  2. Per cultural label (culturally-neutral, persian-centric, english-centric)

Usage:
  python eval/compute_all_metrics_ensemble.py --data_dir data/results/hard/all_templates --fa_prefix evaluate --en_prefix evaluate

"""

import json
import os
import argparse
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict


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


def get_correct_key(item, lang):
    """Get the correct answer as it appears in item['scores']."""
    if lang == "en":
        return item["objects_en"][0]
    else:
        return item["objects"][0]


def get_candidate_keys(item, lang):
    """
    Get all candidate keys in canonical order: [correct, distractor1, ..., distractor4].
    """
    if lang == "en":
        return [item["objects_en"][0]] + list(item["object_candidates_en"])
    else:
        return [item["objects"][0]] + list(item["object_candidates"])


def get_score_vector(item, lang):
    """
    Get the score vector over candidates in canonical order.
    Uses the aggregated 'scores' field (= mean log-prob across 5 templates).
    """
    keys = get_candidate_keys(item, lang)
    scores = []
    for k in keys:
        if k not in item["scores"]:
            return None
        scores.append(item["scores"][k])
    return scores


def get_rank_of_correct(item, lang):
    """
    Get the rank (1-indexed) of the correct answer among all candidates.
    Rank 1 = highest score.
    """
    correct = get_correct_key(item, lang)
    sorted_candidates = sorted(item["scores"].items(), key=lambda x: x[1], reverse=True)
    for rank, (candidate, score) in enumerate(sorted_candidates, start=1):
        if candidate == correct:
            return rank
    return len(item["scores"])


# ═══════════════════════════════════════════════════════════════════════════
# RankC computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_rankc(fa_items, en_items):
    """
    Compute RankC: mean Spearman rank correlation of candidate score vectors
    across FA and EN, averaged over all items.
    """
    per_item_corr = []

    for i in range(len(fa_items)):
        fa_scores = get_score_vector(fa_items[i], "fa")
        en_scores = get_score_vector(en_items[i], "en")

        if fa_scores is None or en_scores is None:
            continue
        if len(fa_scores) != len(en_scores):
            continue
        if len(fa_scores) < 3:
            continue

        corr, _ = spearmanr(fa_scores, en_scores)

        if np.isnan(corr):
            continue

        per_item_corr.append(corr)

    rankc = np.mean(per_item_corr) if per_item_corr else float("nan")
    return rankc, per_item_corr, len(per_item_corr)


# ═══════════════════════════════════════════════════════════════════════════
# All metrics for a subset of items
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(fa_items, en_items):
    """
    Compute all metrics for a list of (fa_item, en_item) pairs.
    Items should already be filtered by label if needed.
    """
    N = len(fa_items)
    if N == 0:
        return None

    assert len(en_items) == N, f"FA ({N}) and EN ({len(en_items)}) have different lengths"

    # Binary correctness arrays
    fa_correct = np.array([item["correct"] for item in fa_items], dtype=bool)
    en_correct = np.array([item["correct"] for item in en_items], dtype=bool)

    # ── Accuracy ──
    acc_fa = fa_correct.mean()
    acc_en = en_correct.mean()

    # ── Binary Consistency ──
    both = (fa_correct & en_correct).mean()       # = Overall
    en_only = (~fa_correct & en_correct).mean()
    fa_only = (fa_correct & ~en_correct).mean()
    neither = (~fa_correct & ~en_correct).mean()
    consist = (fa_correct == en_correct).mean()

    # ── Transfer & Non-Transfer (EN = source, FA = target) ──
    n_en_correct = en_correct.sum()
    n_en_incorrect = (~en_correct).sum()

    if n_en_correct > 0:
        transfer = (fa_correct & en_correct).sum() / n_en_correct  # P(FA correct | EN correct)
    else:
        transfer = float("nan")

    if n_en_incorrect > 0:
        non_transfer = (fa_correct & ~en_correct).sum() / n_en_incorrect  # P(FA correct | EN incorrect)
    else:
        non_transfer = float("nan")

    # ── MRR ──
    fa_ranks = np.array([get_rank_of_correct(item, "fa") for item in fa_items])
    en_ranks = np.array([get_rank_of_correct(item, "en") for item in en_items])
    mrr_fa = np.mean(1.0 / fa_ranks)
    mrr_en = np.mean(1.0 / en_ranks)
    mrr_min = np.mean(np.minimum(1.0 / fa_ranks, 1.0 / en_ranks))

    # ── RankC ──
    rankc, per_item_corr, n_valid = compute_rankc(fa_items, en_items)

    return {
        "n": N,
        "acc_fa": acc_fa * 100,
        "acc_en": acc_en * 100,
        "both": both * 100,          # Overall
        "en_only": en_only * 100,
        "fa_only": fa_only * 100,
        "neither": neither * 100,
        "consist": consist * 100,
        "transfer": transfer * 100,       # P(FA+ | EN+)
        "non_transfer": non_transfer * 100,  # P(FA+ | EN-)
        "n_en_correct": int(n_en_correct),
        "n_en_incorrect": int(n_en_incorrect),
        "mrr_fa": mrr_fa * 100,
        "mrr_en": mrr_en * 100,
        "mrr_min": mrr_min * 100,
        "rankc": rankc * 100,
        "n_valid_rankc": n_valid,
    }


def filter_by_label(fa_data, en_data, label):
    """Filter FA and EN items by cultural label. Label=None returns all."""
    if label is None:
        return fa_data, en_data
    fa_filtered = []
    en_filtered = []
    for i in range(len(fa_data)):
        # Label should be in EN data (or FA data — they match)
        item_label = en_data[i].get("label", fa_data[i].get("label", None))
        if item_label == label:
            fa_filtered.append(fa_data[i])
            en_filtered.append(en_data[i])
    return fa_filtered, en_filtered


# ═══════════════════════════════════════════════════════════════════════════
# Pretty printing
# ═══════════════════════════════════════════════════════════════════════════

def print_metrics_table(all_results, label_name, models):
    """Print a formatted table for one label subset."""
    print(f"\n{'='*120}")
    print(f"  {label_name}")
    print(f"{'='*120}")
    header = (
        f"{'Model':<42} {'N':>5} {'Acc_EN':>7} {'Acc_FA':>7} "
        f"{'Transf':>7} {'NonTr':>7} {'(n EN+)':>8} {'(n EN-)':>8} "
        f"{'Overall':>8} {'Consist':>8} {'RankC':>7}"
    )
    print(header)
    print("-" * len(header))

    for model in models:
        m = all_results[model]
        if m is None:
            print(f"{model:<42} {'(no data)':>5}")
            continue
        print(
            f"{model:<42} {m['n']:>5} {m['acc_en']:>7.1f} {m['acc_fa']:>7.1f} "
            f"{m['transfer']:>7.1f} {m['non_transfer']:>7.1f} "
            f"{m['n_en_correct']:>8} {m['n_en_incorrect']:>8} "
            f"{m['both']:>8.1f} {m['consist']:>8.1f} {m['rankc']:>7.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-lingual metrics with Transfer/Non-Transfer and per-label breakdown"
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

    # Cultural labels to break down by
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
                en_prefix=args.en_prefix
            )
            fa_subset, en_subset = filter_by_label(fa_data, en_data, label)
            if len(fa_subset) == 0:
                label_results[model] = None
            else:
                label_results[model] = compute_all_metrics(fa_subset, en_subset)
        all_label_results[label_name] = label_results
        print_metrics_table(label_results, label_name, models)

    # ── Optional CSV output ──
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label", "model", "n", "acc_en", "acc_fa",
                "transfer", "non_transfer", "n_en_correct", "n_en_incorrect",
                "overall", "en_only", "fa_only", "neither", "consist",
                "mrr_fa", "mrr_en", "mrr_min", "rankc"
            ])
            for label_name, label_results in all_label_results.items():
                for model in models:
                    m = label_results[model]
                    if m is None:
                        continue
                    writer.writerow([
                        label_name, model, m["n"],
                        f"{m['acc_en']:.2f}", f"{m['acc_fa']:.2f}",
                        f"{m['transfer']:.2f}", f"{m['non_transfer']:.2f}",
                        m["n_en_correct"], m["n_en_incorrect"],
                        f"{m['both']:.2f}", f"{m['en_only']:.2f}",
                        f"{m['fa_only']:.2f}", f"{m['neither']:.2f}",
                        f"{m['consist']:.2f}",
                        f"{m['mrr_fa']:.4f}", f"{m['mrr_en']:.4f}",
                        f"{m['mrr_min']:.4f}", f"{m['rankc']:.4f}",
                    ])
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()