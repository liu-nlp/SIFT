#!/bin/bash
# Evaluate each model on the English and Persian versions of the probe set,
# then aggregate the metrics.
set -e

MODELS=(
    "liu-nlp/hyperllama-572m-english-1x-cloned"   # EN-Base
    "rominaoji/base-persian-only"                 # FA-Base
    "rominaoji/cpt-full-persian"                  # CPT-Full
    "rominaoji/cpt-subj-obj-removed-persian"      # CPT-CoocRM
    "rominaoji/cpt-subj-removed-persian"          # CPT-SubjRM
    "rominaoji/cpt-subj-replaced-persian"         # CPT-SubjRP
    "rominaoji/persian_keep_fact"                 # CPT-KeepAll
)

LEVEL="hard"   # hard | random
DATA="data/facts/triples-500-${LEVEL}.json"
OUT_DIR="data/results/${LEVEL}/all_templates_qa"

for MODEL in "${MODELS[@]}"; do
    MODEL_TAG=$(basename "$MODEL")

    # English
    python eval/evaluate_final.py \
        --data            "$DATA" \
        --templates       data/facts/templates_qa_en.json \
        --model           "$MODEL" \
        --output          "${OUT_DIR}/evaluate_${MODEL_TAG}_all_relations_en.json" \
        --num-candidates  4 \
        --language        en \
        --subject-variant first \
        --level           "$LEVEL"

    # Persian
    python eval/evaluate_final.py \
        --data            "$DATA" \
        --templates       data/facts/improved_templates_fa_v2_qa.json \
        --model           "$MODEL" \
        --output          "${OUT_DIR}/evaluate_${MODEL_TAG}_all_relations_fa.json" \
        --num-candidates  4 \
        --language        fa \
        --subject-variant first \
        --level           "$LEVEL"
done

# Aggregate metrics (accuracy, consistency, transfer, MRR, RankC)
python eval/compute_all_metrics_ensemble.py --data_dir "$OUT_DIR" --fa_prefix evaluate --en_prefix evaluate
python eval/compute_all_metrics_mean_std.py --data_dir "$OUT_DIR" --fa_prefix evaluate --en_prefix evaluate
