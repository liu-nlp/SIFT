# Probing Factual Knowledge Transfer with Training Data Interventions

Code for the data interventions, training configurations and evaluation used in the EMNLP 2026 paper
[*Probing Factual Knowledge Transfer with Training Data Interventions*](https://aclanthology.org/2026.emnlp-main.2445/).

## Repository layout

```
preprocess/
  preprocess_coocRM.py               # CoocRM: remove sentences where a subject and its object co-occur
  preprocess_coocRM_with_window.py   # CoocRM + removal of ±W neighbouring sentences (W=0 == CoocRM)
  preprocess_remove_all.py           # SubjRM: remove every sentence mentioning a subject
  preprocess_remove_replace.py       # Remove 80% / replace 20% of subject sentences with templates
  preprocess_keep_all.py             # Keep-all control: keep fact-bearing documents, remove a matched
                                     #   random share (~8.7%) of sentences from the rest
eval/
  evaluate_final.py                  # multiple-choice factual probing (EN / FA), template-averaged
  compute_all_metrics_ensemble.py    # accuracy, consistency, transfer, MRR, RankC (template ensemble)
  compute_all_metrics_mean_std.py    # same metrics, reported as mean ± std over templates
configs/                             # nanotron continued-pretraining configs, one per condition
scripts/
  run_preprocess.sh                  # builds all filtered corpora
  run_eval.sh                        # runs evaluation + metric aggregation
```

## Setup

```bash
pip install -r requirements.txt
```

The preprocessing scripts stream the Persian split (`fas_Arab`) of
[FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).
If you need authenticated Hub access, export `HF_TOKEN` before running.

## 1. Building the filtered corpora

```bash
bash scripts/run_preprocess.sh
```

Every script writes `filtered/` (the cleaned documents used for training) and
`removed/` (the removed sentences with metadata) into its output directory.
Use `--mode subset --sample-size N` for a quick test on a sample.

## 2. Continued pretraining

`configs/` holds the [nanotron](https://github.com/huggingface/nanotron) configs used for each condition:

| Config | Condition |
|---|---|
| `persian_subj_obj_removed.yaml` | Co-occurrence removal (CoocRM) |
| `persian_subj_removed.yaml` | Subject removal (SubjRM) |
| `persian_subj_replaced_removed.yaml` | Subject remove + replace |
| `persian_keep_fact.yaml` | Keep all (control) |

Before training, tokenize the `filtered/` output into a nanoset and replace the
placeholders (`<PATH_TO_ENGLISH_BASE_CHECKPOINT>`, `<PATH_TO_PERSIAN_REPLAY_NANOSET>`,
`<PATH_TO_CODE_NANOSET>`, `<PATH_TO_LLAMA_3.3_TOKENIZER>`) with your own paths.

## 3. Evaluation

```bash
bash scripts/run_eval.sh
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{oji-etal-2026-probing,
    title = "Probing Factual Knowledge Transfer with Training Data Interventions",
    author = "Oji, Romina  and
      Braun, Marc  and
      Bollmann, Marcel  and
      Kuhlmann, Marco  and
      Kunz, Jenny",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2445/",
    doi = "10.18653/2026.emnlp-main.2445",
    pages = "55474--55492",
    ISBN = "979-8-89176-459-0"
}
```
