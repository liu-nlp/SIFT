#!/bin/bash
# Build the four filtered FineWeb2 (fas_Arab) corpora used in the paper.
# Set HF_TOKEN in your environment beforehand if you need authenticated access:
#   export HF_TOKEN=<your_token>
set -e

PAIRS=data/facts/triples-500-improved.json
SUBJ_TEMPLATES=data/facts/subject_templates_100.json
WORKERS=32
CACHE=data/cache

# 1) Co-occurrence removal (CoocRM): drop sentences where subject AND object co-occur
python preprocess/preprocess_coocRM.py \
    --pairs     $PAIRS \
    --output    data/subj_obj_removed \
    --mode      full \
    --workers   $WORKERS \
    --cache-dir $CACHE

#    Optional: also remove W neighbouring sentences (--window-size 0 == the script above)
# python preprocess/preprocess_coocRM_with_window.py \
#     --pairs data/facts/triples-500-hard.json --output data/subj_obj_removed_window_1 \
#     --mode full --window-size 1 --workers $WORKERS --cache-dir $CACHE

# 2) Subject removal (SubjRM): drop every sentence mentioning a subject
python preprocess/preprocess_remove_all.py \
    --pairs         $PAIRS \
    --output        data/subj_removed \
    --sentence-mode remove \
    --mode          full \
    --workers       $WORKERS \
    --cache-dir     $CACHE

# 3) Remove + replace: 80% of subject sentences removed, 20% replaced by templates.
#    Uses the per-subject match counts produced by step 2.
python preprocess/preprocess_remove_replace.py \
    --pairs      $PAIRS \
    --templates  $SUBJ_TEMPLATES \
    --counts-dir data/subj_removed/match_counts.json \
    --output     data/subjects_removed_replaced \
    --mode       full \
    --workers    $WORKERS \
    --cache-dir  $CACHE

# 4) Keep all (control): keep every fact-bearing document, randomly remove the
#    same share of sentences (~8.7%) from the remaining documents.
#    4a. calibrate the removal probability on a sample
python preprocess/preprocess_keep_all.py \
    --pairs          $PAIRS \
    --output         data/kept_facts_dataset \
    --mode           subset \
    --sample-size    100000 \
    --target-percent 8.7 \
    --cache-dir      $CACHE
#    4b. full run using the calibrated probability
python preprocess/preprocess_keep_all.py \
    --pairs     $PAIRS \
    --output    data/kept_facts_dataset \
    --mode      full \
    --workers   16 \
    --cache-dir $CACHE
