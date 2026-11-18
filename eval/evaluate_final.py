#!/usr/bin/env python3
"""
Evaluation with template averaging and per-template std reporting.

Scoring:
  - Causal LM: full-sentence scoring (handles both QA and declarative templates)
  - Masked LM: standard masked LM scoring

Output:
  - Per-relation accuracy (averaged across all templates)
  - Per-template accuracies and std at the end
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

_CAUSAL_MODEL_TYPES = {
    'llama', 'mistral', 'mixtral', 'gemma', 'gemma2', 'gemma3',
    'gpt2', 'gpt_neo', 'gpt_neox', 'gpt_bigcode', 'gptj',
    'falcon', 'mpt', 'bloom', 'opt', 'qwen2', 'qwen2_moe',
    'stablelm', 'starcoder2', 'phi', 'phi3', 'cohere',
    'olmo', 'olmoe', 'smollm', 'smollm2', 'hyperllama',
}


def _is_causal_model(config):
    architectures = getattr(config, 'architectures', None) or []
    if any('CausalLM' in arch for arch in architectures):
        return True
    return getattr(config, 'model_type', '') in _CAUSAL_MODEL_TYPES


# ============================================================================
# SCORING
# ============================================================================

def score_candidate_masked(model, tokenizer, prompt, candidate, device):
    """Masked LM scoring: P(candidate tokens | rest of sentence)."""
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
    n = len(cand_ids)
    if n == 0:
        return float('-inf')

    masked_prompt = prompt.replace(
        '<mask>', ' '.join([tokenizer.mask_token] * n)
    )
    inputs = tokenizer(masked_prompt, return_tensors='pt',
                       truncation=True, max_length=512).to(device)
    mask_pos = (inputs.input_ids == tokenizer.mask_token_id) \
        .nonzero(as_tuple=True)[1]
    if len(mask_pos) != n:
        return float('-inf')

    with torch.no_grad():
        logits = model(**inputs).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    score = sum(log_probs[pos, tok_id].item()
                for pos, tok_id in zip(mask_pos, cand_ids))
    return score / n


# def score_candidate_causal_full(model, tokenizer, prompt, candidate, device):
#     """
#     Causal LM whole-sentence scoring.

#     Computes the length-normalised log-probability of the filled sentence:
#         log P(sentence) = sum over all positions of log P(token_i | token_<i)
#         score = log P(sentence) / |sentence|

#     Every token in the filled sentence is scored, including the prefix.
#     The prefix probability is identical across candidates so it does not
#     affect ranking, but including it makes the metric a true sentence
#     probability for clarity in reporting.
#     """
#     full_sentence = prompt.replace('<mask>', candidate)
#     full_ids = tokenizer.encode(full_sentence, add_special_tokens=True,
#                                 truncation=True, max_length=512)

#     if len(full_ids) < 2:
#         return float('-inf')

#     input_ids = torch.tensor([full_ids], device=device)
#     with torch.no_grad():
#         logits = model(input_ids=input_ids).logits[0]
#     log_probs = F.log_softmax(logits, dim=-1)

#     # Score every next-token prediction across the entire sentence.
#     # Position pos predicts token at pos+1, so we iterate to len-2.
#     score = 0.0
#     n_scored = 0
#     for pos in range(len(full_ids) - 1):
#         tok_id = full_ids[pos + 1]
#         score += log_probs[pos, tok_id].item()
#         n_scored += 1

#     return score / n_scored if n_scored > 0 else float('-inf')

def score_candidate_causal_full(
    model, tokenizer, prompt, candidate, device
) -> float:
    # Fill in the candidate to get a complete sentence
    full_sentence = prompt.replace('<mask>', candidate)
    prefix = prompt.split('<mask>', 1)[0]

    full_ids = tokenizer.encode(full_sentence, add_special_tokens=True,
                                truncation=True, max_length=512)
    prefix_ids = tokenizer.encode(prefix.rstrip(), add_special_tokens=True)
    n_prefix = len(prefix_ids)

    # Candidate token span in the full sentence
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
    n_cand = len(cand_ids)
    if n_cand == 0:
        return float('-inf')

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]

    log_probs = F.log_softmax(logits, dim=-1)

    # Score tokens from start of candidate to end of sentence
    # (prefix probability is constant across candidates, so skip it)
    score = 0.0
    n_scored = 0
    for pos in range(n_prefix - 1, len(full_ids) - 1):
        tok_id = full_ids[pos + 1]
        score += log_probs[pos, tok_id].item()
        n_scored += 1

    return score / n_scored if n_scored > 0 else float('-inf')



# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(data, relation_templates, model, tokenizer, device, score_fn,
             subject_variant='first', num_candidates=None, language='fa'):
    results = []

    for entry in tqdm(data, desc='Scoring entries'):
        relation = entry['relation']
        templates = relation_templates.get(relation)
        if not templates:
            continue

        if language == 'en':
            gold_set = set(entry['objects_en'])
            gold = entry['objects_en'][0]
            candidates = entry.get('object_candidates_en', [])
            subject = entry['subject_en']
        else:
            gold_set = set(entry['objects'])
            gold = entry['objects'][0]
            candidates = entry.get('object_candidates', [])
            subjects = entry.get('subjects') or [entry['subject']]
            if subject_variant == 'first':
                subject = subjects[0]
            elif subject_variant == 'second':
                subject = subjects[1] if len(subjects) > 1 else subjects[0]
            else:
                subject = random.choice(subjects)

        if num_candidates is not None and len(candidates) > num_candidates:
            candidates = random.sample(candidates, num_candidates)

        choices = [gold] + candidates
        seen, unique_choices = set(), []
        for c in choices:
            if c not in seen:
                unique_choices.append(c)
                seen.add(c)
        choices = unique_choices

        # Score with each template separately
        per_template_scores = []
        per_template_correct = []

        for template in templates:
            prompt = template.replace('<subject>', subject)
            t_scores = {c: score_fn(model, tokenizer, prompt, c, device)
                        for c in choices}
            t_predicted = max(t_scores, key=t_scores.get)
            per_template_scores.append(t_scores)
            per_template_correct.append(t_predicted in gold_set)

        # Aggregate: mean score across templates
        mean_scores = {
            c: sum(t[c] for t in per_template_scores) / len(per_template_scores)
            for c in choices
        }
        predicted = max(mean_scores, key=mean_scores.get)
        is_correct = predicted in gold_set

        results.append({
            **entry,
            'predicted':            predicted,
            'scores':               mean_scores,
            'per_template_scores':  per_template_scores,
            'per_template_correct': per_template_correct,
            'correct':              is_correct,
        })

    return results


# ============================================================================
# REPORTING
# ============================================================================

def print_report(results):
    by_relation = defaultdict(list)
    for r in results:
        by_relation[r['relation']].append(
            (r['correct'], r['per_template_correct'])
        )

    n_templates = len(results[0]['per_template_correct']) if results else 1

    col_w = max(len(rel) for rel in by_relation) + 2
    header = (f"{'Relation':<{col_w}}  {'Correct':>7}  {'Total':>5}"
              f"  {'Acc (mean)':>10}  {'Std':>6}")
    print('\n' + header)
    print('-' * len(header))

    total_correct = total_count = 0
    overall_per_template = [0] * n_templates
    overall_per_template_total = [0] * n_templates

    for rel in by_relation:
        entries = by_relation[rel]
        n_total = len(entries)

        n_correct_mean = sum(e[0] for e in entries)
        acc_mean = 100.0 * n_correct_mean / n_total if n_total else 0.0

        # Per-template accuracies for THIS relation
        template_accs = []
        for t in range(n_templates):
            t_correct = sum(e[1][t] for e in entries if t < len(e[1]))
            template_accs.append(100.0 * t_correct / n_total)
            # Aggregate for overall computation
            overall_per_template[t] += t_correct
            overall_per_template_total[t] += n_total

        if len(template_accs) > 1:
            mean_t = sum(template_accs) / len(template_accs)
            var_t = sum((x - mean_t) ** 2 for x in template_accs) / (len(template_accs) - 1)
            std_t = math.sqrt(var_t)
        else:
            std_t = 0.0

        print(f"{rel:<{col_w}}  {n_correct_mean:>7}  {n_total:>5}"
              f"  {acc_mean:>9.1f}%  {std_t:>5.1f}")

        total_correct += n_correct_mean
        total_count += n_total

    print('-' * len(header))
    overall = 100.0 * total_correct / total_count if total_count else 0.0
    print(f"{'Overall':<{col_w}}  {total_correct:>7}  {total_count:>5}"
          f"  {overall:>9.1f}%")

    # ====================================================================
    # PER-TEMPLATE SUMMARY (the format you asked for)
    # ====================================================================
    overall_template_accs = [
        100.0 * c / t if t > 0 else 0.0
        for c, t in zip(overall_per_template, overall_per_template_total)
    ]
    if len(overall_template_accs) > 1:
        mean_o = sum(overall_template_accs) / len(overall_template_accs)
        var_o = sum((x - mean_o) ** 2 for x in overall_template_accs) / (len(overall_template_accs) - 1)
        std_o = math.sqrt(var_o)
    else:
        std_o = 0.0

    print()
    formatted = "[" + ", ".join(f"{v:.1f}" for v in overall_template_accs) + "]"
    print(f"per-template accs = {formatted}")
    print(f"std = {std_o:.2f}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/facts/triples_en_fa.json')
    parser.add_argument('--templates', default='data/facts/relation_templates.json')
    parser.add_argument('--model', default='bert-base-multilingual-cased')
    parser.add_argument('--num-candidates', type=int, default=None)
    parser.add_argument('--subject-variant',
                        choices=['first', 'second', 'random'], default='first')
    parser.add_argument('--output', default=None)
    parser.add_argument('--language', choices=['fa', 'en'], default='fa')
    parser.add_argument('--level', choices=['random', 'hard'], default='hard')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    random.seed(args.seed)

    if args.language == 'en' and args.templates == 'data/facts/relation_templates.json':
        args.templates = 'data/facts/relation_templates_en.json'

    print(f'Loading data from {args.data}')
    with open(args.data, encoding='utf-8') as f:
        data = json.load(f)
    print(f'  {len(data)} entries loaded')

    print(f'Loading templates from {args.templates}')
    with open(args.templates, encoding='utf-8') as f:
        relation_templates = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model "{args.model}" on {device}')

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_causal = _is_causal_model(config)
    print(f'  Model type: {config.model_type} → {"causal" if is_causal else "masked"} LM scoring')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if is_causal:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)
        score_fn = score_candidate_causal_full
        print('  Using whole-sentence scoring (length-normalised log-probability)')
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True).to(device)
        score_fn = score_candidate_masked
    model.eval()

    results = evaluate(data, relation_templates, model, tokenizer, device, score_fn,
                       subject_variant=args.subject_variant,
                       num_candidates=args.num_candidates,
                       language=args.language)
    print_report(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _clean(v):
            if isinstance(v, float) and (not (v == v) or abs(v) == float('inf')):
                return None
            return v

        serialisable = []
        for r in results:
            entry = {k: v for k, v in r.items()
                     if k not in ('scores', 'per_template_scores')}
            entry['scores'] = {c: _clean(s) for c, s in r['scores'].items()}
            entry['per_template_scores'] = [
                {c: _clean(s) for c, s in t.items()}
                for t in r['per_template_scores']
            ]
            serialisable.append(entry)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        print(f'Results written to {out_path}')


if __name__ == '__main__':
    main()
