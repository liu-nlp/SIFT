#!/usr/bin/env python3
"""
preprocess_keep_all.py

Document-level SIFT protection with calibrated random removal from the
FineWeb2 Persian dataset.

Unlike preprocess_remove_all.py (which removes any *sentence* containing a
known subject, regardless of the rest of the document), this script:

  1. Treats a whole document as PROTECTED (kept 100% unchanged) if it
     contains any known subject variant OR any true object variant
     anywhere in its text (object_candidates / distractors do not count).
  2. From the remaining "safe" documents (zero subject/object signal),
     removes a random sample of sentences, calibrated so the overall
     sentence-removal rate across the corpus lands close to a target
     percentage (default 8.7%, to match preprocess_remove_all.py's output).

Workflow (two steps):

  # 1. Calibrate: estimate the removal probability from a large sample.
  python preprocess_keep_all.py \\
      --pairs         data/facts/triples-en-fa.json \\
      --output        data/kept_facts_dataset \\
      --mode          subset \\
      --sample-size   100000 \\
      --target-percent 8.7
  # -> prints + writes data/kept_facts_dataset/calibration.json

  # 2. Full run: applies the calibrated probability across all shards.
  HF_TOKEN=hf_xxx python preprocess_keep_all.py \\
      --pairs         data/facts/triples-en-fa.json \\
      --output        data/kept_facts_dataset \\
      --mode          full \\
      --workers       16

Outputs (same layout as preprocess_remove_all.py):

  - filtered/  : documents with random sentences removed (protected docs
                 are byte-identical to the input)
  - removed/   : extracted sentences with metadata
  - counts/    : per-shard counts (full mode)
  - calibration.json    : written by subset mode
  - removal_counts.json : merged counts + realized removal % (full mode)
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# ParsiNorm import (local copy)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "ParsiNorm"))
from parsinorm import Tokenizer as _ParsiNormTokenizer

# ---------------------------------------------------------------------------
# Lazy imports — only loaded when actually needed.
# ---------------------------------------------------------------------------
_pa = None
_pq = None


def _ensure_pyarrow():
    global _pa, _pq
    if _pa is None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        _pa = pa
        _pq = pq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_REPO     = "HuggingFaceFW/fineweb-2"
LANGUAGE_CODE    = "fas_Arab"
PROXIMITY_WINDOW = 300   # max chars for a sentence before chunking
SHARD_TIMEOUT    = 2 * 3600  # 2 hours per shard


# ---------------------------------------------------------------------------
# Section 1: Normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(
    r'[،؛؟۔«»(){}\[\]/\\"\'`|^~*+=#@$%&'   # Persian / Arabic punctuation
    r',;?!.()\[\]{}/:–—\-'                    # ASCII punctuation
    r'‘’“”‹›]'  # typographic quotes
)


def normalize_for_matching(text: str) -> str:
    """Normalise text for matching purposes only -- never written to output."""
    text = text.replace('‌', ' ')   # ZWNJ / half-space
    text = text.replace('​', ' ')   # zero-width space
    text = text.replace(' ', ' ')   # non-breaking space
    text = _PUNCT_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Section 2: Entity Matching (Aho-Corasick over subjects + true objects)
# ---------------------------------------------------------------------------

try:
    import ahocorasick
    _HAS_AHOCORASICK = True
except ImportError:
    _HAS_AHOCORASICK = False


class EntityMatcher:
    """Load triples JSON, build an Aho-Corasick automaton over every known
    subject variant AND every true object variant (object_candidates /
    distractors are ignored).

    A document is "protected" if any of these terms appears anywhere in its
    text — matching is done once over the whole (normalised) document, not
    per sentence.
    """

    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # entries[i] = {relation, subject_en, subject}
        self.entries = []
        # norm_term -> (entry_idx, term_type, original_term)
        self._term_to_entry: dict[str, tuple] = {}

        all_terms: set[str] = set()

        for item in data:
            subjects = item.get('subjects', []) or []
            objects  = item.get('objects', []) or []
            if not subjects and not objects:
                continue

            entry_idx = len(self.entries)
            self.entries.append({
                'relation':   item.get('relation', ''),
                'subject_en': item.get('subject_en', ''),
                'subject':    item.get('subject', subjects[0] if subjects else ''),
            })

            for term_type, terms in (('subject', subjects), ('object', objects)):
                for term in {s for s in terms if s and s.strip()}:
                    norm = normalize_for_matching(term)
                    if not norm:
                        continue
                    all_terms.add(norm)
                    # First entry/term wins on collision — matches the
                    # first-match semantics used at query time.
                    self._term_to_entry.setdefault(norm, (entry_idx, term_type, term))

        # Build Aho-Corasick automaton over all subject + object terms
        if _HAS_AHOCORASICK and all_terms:
            self._automaton = ahocorasick.Automaton()
            for term in all_terms:
                self._automaton.add_word(term, term)
            self._automaton.make_automaton()
            self._use_ac = True
        else:
            self._use_ac = False
            self._all_norm_terms = sorted(all_terms, key=len, reverse=True)

    def find_match(self, text: str):
        """Return (True, term, term_type, relation) or (False, None, None, None).

        *text* is the original (un-normalised) text — normalisation is
        applied internally. Intended to be called once on a whole document.
        """
        norm = ' ' + normalize_for_matching(text) + ' '

        if self._use_ac:
            return self._find_match_ac(norm)
        else:
            return self._find_match_linear(norm)

    def _find_match_ac(self, norm: str):
        for end_idx, term in self._automaton.iter(norm):
            start_idx = end_idx - len(term) + 1
            if norm[start_idx - 1] == ' ' and (end_idx + 1 >= len(norm) or norm[end_idx + 1] == ' '):
                entry_idx, term_type, orig_term = self._term_to_entry[term]
                entry = self.entries[entry_idx]
                return True, orig_term, term_type, entry['relation']
        return False, None, None, None

    def _find_match_linear(self, norm: str):
        for term in self._all_norm_terms:
            if ' ' + term + ' ' in norm:
                entry_idx, term_type, orig_term = self._term_to_entry[term]
                entry = self.entries[entry_idx]
                return True, orig_term, term_type, entry['relation']
        return False, None, None, None


# ---------------------------------------------------------------------------
# Section 3: Sentence Splitting (ParsiNorm)
# ---------------------------------------------------------------------------

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _ParsiNormTokenizer()
    return _tokenizer


def split_sentences(text: str) -> list[str]:
    """Split a single line/paragraph into sentences using ParsiNorm."""
    if not text.strip():
        return []
    try:
        return _get_tokenizer().sentence_tokenize(text, verb_seperator=False)
    except Exception:
        return [text]


def chunk_sentence(sentence: str, max_chars: int = PROXIMITY_WINDOW) -> list[str]:
    """Split a long sentence into chunks at word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]

    chunks = []
    remaining = sentence
    while len(remaining) > max_chars:
        split_at = remaining.rfind(' ', 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _count_chunks(text: str) -> int:
    """Count sentence/chunks in a document without building any output."""
    n = 0
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        for sent in split_sentences(stripped):
            chunks = chunk_sentence(sent) if len(sent) > PROXIMITY_WINDOW else [sent]
            n += len(chunks)
    return n


# ---------------------------------------------------------------------------
# Section 4: I/O Helpers
# ---------------------------------------------------------------------------

def download_all_shards(cache_dir: Path, n_threads: int = 8) -> list[str]:
    """List Persian Parquet shards from HF Hub and download them."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from tqdm import tqdm

    all_files = list(list_repo_files(DATASET_REPO, repo_type="dataset"))
    fa_files  = sorted(
        f for f in all_files
        if f.startswith(f"data/{LANGUAGE_CODE}/") and f.endswith(".parquet")
    )
    print(f"Found {len(fa_files)} Persian Parquet shards")

    local_paths = [None] * len(fa_files)
    fname_to_idx = {f: i for i, f in enumerate(fa_files)}

    def _download_one(fname):
        local = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=fname,
            cache_dir=str(cache_dir),
        )
        return fname, local

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(_download_one, f): f for f in fa_files}
        with tqdm(total=len(fa_files), desc="Downloading shards", unit="shard") as pbar:
            for future in as_completed(futures):
                fname, local = future.result()
                local_paths[fname_to_idx[fname]] = local
                pbar.update(1)

    return local_paths


def iter_parquet_rows(shard_path: str, batch_size: int = 5000):
    """Yield row dicts from a Parquet file using batched reading."""
    _ensure_pyarrow()
    pfile = _pq.ParquetFile(shard_path)
    for batch in pfile.iter_batches(batch_size=batch_size):
        columns  = batch.schema.names
        col_data = {col: batch.column(col).to_pylist() for col in columns}
        for i in range(batch.num_rows):
            yield {col: col_data[col][i] for col in columns}


# ---------------------------------------------------------------------------
# Section 5: Document Processing
# ---------------------------------------------------------------------------

def process_document(row: dict, matcher: EntityMatcher, removal_prob: float,
                      rng: random.Random) -> tuple:
    """Filter one FineWeb2 document under document-level SIFT protection.

    Returns (filtered_text | None, removed_list, protected, protect_info,
    n_sentences):
      - protected    : bool, whether the doc matched a subject/object
      - protect_info : {'term','term_type','relation'} if protected else None
      - n_sentences  : total sentence/chunk count in the document (for
                       accurate global removal-rate accounting; counted for
                       both protected and unprotected documents)
    """
    text = row.get('text', '') or ''
    if not text.strip():
        return None, [], False, None, 0

    doc_id = row.get('id',  '')
    url    = row.get('url', '')

    matched, term, term_type, relation = matcher.find_match(text)

    if matched:
        n_sentences = _count_chunks(text)
        protect_info = {'term': term, 'term_type': term_type, 'relation': relation}
        return text, [], True, protect_info, n_sentences

    lines        = text.split('\n')
    kept_lines   = []
    removed_list = []
    n_sentences  = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append('')
            continue

        sentences  = split_sentences(stripped)
        kept_sents = []

        for sent in sentences:
            chunks = chunk_sentence(sent) if len(sent) > PROXIMITY_WINDOW else [sent]

            for chunk in chunks:
                n_sentences += 1
                if rng.random() < removal_prob:
                    removed_list.append({
                        'doc_id':   doc_id,
                        'url':      url,
                        'sentence': chunk,
                        'reason':   'random_unrelated_sample',
                    })
                else:
                    kept_sents.append(chunk)

        if kept_sents:
            kept_lines.append(' '.join(kept_sents))

    while kept_lines and not kept_lines[-1].strip():
        kept_lines.pop()

    filtered_text = '\n'.join(kept_lines).strip()
    return (filtered_text if filtered_text else None), removed_list, False, None, n_sentences


# ---------------------------------------------------------------------------
# Section 6: Shard Worker
# ---------------------------------------------------------------------------

def run_worker(shard_path: str, pairs_path: str, output_dir: str,
               shard_idx: int, removal_prob: float, seed: int) -> None:
    """Process one Parquet shard end-to-end. Called as a subprocess."""
    out = Path(output_dir)
    filtered_path = out / 'filtered' / f'shard_{shard_idx:04d}.jsonl'
    removed_path  = out / 'removed'  / f'shard_{shard_idx:04d}.jsonl'
    counts_path   = out / 'counts'   / f'shard_{shard_idx:04d}_counts.json'

    if filtered_path.exists() and removed_path.exists() and counts_path.exists():
        print(f"[Shard {shard_idx:04d}] SKIP (already done)", flush=True)
        return

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)

    matcher = EntityMatcher(pairs_path)
    print(f"[Shard {shard_idx:04d}] Loaded {len(matcher.entries)} entries, "
          f"AC={'yes' if matcher._use_ac else 'no'}, removal_prob={removal_prob}", flush=True)

    rng = random.Random(f"{seed}-{shard_idx}")

    n_rows = n_kept = n_protected_docs = 0
    n_removed_sentences = n_sentences_total = n_protected_sentences = 0
    entity_counts = Counter()

    with open(filtered_path, 'w', encoding='utf-8') as ff, \
         open(removed_path,  'w', encoding='utf-8') as rf:

        for row in iter_parquet_rows(shard_path):
            n_rows += 1
            ftext, removed, protected, protect_info, n_sent = process_document(
                row, matcher, removal_prob, rng)

            n_sentences_total += n_sent

            if protected:
                n_protected_docs += 1
                n_protected_sentences += n_sent
                entity_counts[protect_info['term']] += 1

            if ftext is not None:
                row['text'] = ftext
                ff.write(json.dumps(row, ensure_ascii=False) + '\n')
                n_kept += 1

            for r in removed:
                rf.write(json.dumps(r, ensure_ascii=False) + '\n')
            n_removed_sentences += len(removed)

    counts = {
        'n_rows':                n_rows,
        'n_kept':                n_kept,
        'n_protected_docs':      n_protected_docs,
        'n_protected_sentences': n_protected_sentences,
        'n_removed_sentences':   n_removed_sentences,
        'n_sentences_total':     n_sentences_total,
        'by_entity':             dict(entity_counts.most_common()),
    }
    with open(counts_path, 'w', encoding='utf-8') as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    print(f"[Shard {shard_idx:04d}] DONE  rows={n_rows:,}  kept={n_kept:,}  "
          f"protected_docs={n_protected_docs:,}  removed_sents={n_removed_sentences:,}", flush=True)


# ---------------------------------------------------------------------------
# Section 7: CLI — Subset Mode (calibration + sample QA)
# ---------------------------------------------------------------------------

def run_subset(args: argparse.Namespace) -> None:
    """Stream first N documents, estimate the removal probability needed to
    hit --target-percent, and (if --removal-prob is also given) apply it to
    the sample so the output can be inspected."""
    from datasets import load_dataset
    from tqdm import tqdm

    matcher = EntityMatcher(args.pairs)
    print(f"Loaded {len(matcher.entries)} entries from {args.pairs}")
    print(f"Aho-Corasick: {'enabled' if matcher._use_ac else 'disabled (install pyahocorasick)'}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_removal = args.removal_prob is not None
    rng = random.Random(f"{args.seed}-subset")

    ff = rf = None
    if apply_removal:
        (out_dir / 'filtered').mkdir(parents=True, exist_ok=True)
        (out_dir / 'removed').mkdir(parents=True, exist_ok=True)
        ff = open(out_dir / 'filtered' / 'sample.jsonl', 'w', encoding='utf-8')
        rf = open(out_dir / 'removed'  / 'sample.jsonl', 'w', encoding='utf-8')

    print(f"Streaming first {args.sample_size} documents "
          f"from {DATASET_REPO} (lang={LANGUAGE_CODE}) ...")

    dataset = load_dataset(
        DATASET_REPO,
        name=LANGUAGE_CODE,
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    total = kept = n_protected_docs = 0
    n_removed_sentences = n_sentences_total = n_protected_sentences = 0
    entity_counts = Counter()

    with tqdm(total=args.sample_size, desc="Documents",
              unit="doc", dynamic_ncols=True) as pbar:
        for record in dataset:
            if total >= args.sample_size:
                break

            ftext, removed, protected, protect_info, n_sent = process_document(
                record, matcher, args.removal_prob or 0.0, rng)

            n_sentences_total += n_sent
            if protected:
                n_protected_docs += 1
                n_protected_sentences += n_sent
                entity_counts[protect_info['term']] += 1

            if apply_removal:
                if ftext is not None:
                    record['text'] = ftext
                    ff.write(json.dumps(record, ensure_ascii=False) + '\n')
                    kept += 1
                for r in removed:
                    rf.write(json.dumps(r, ensure_ascii=False) + '\n')
            n_removed_sentences += len(removed)

            total += 1
            pbar.update(1)
            pbar.set_postfix(protected=n_protected_docs, sentences=n_sentences_total)

    if apply_removal:
        ff.close()
        rf.close()

    n_unprotected_sentences = n_sentences_total - n_protected_sentences
    target_fraction = args.target_percent / 100.0

    recommended_prob = None
    if n_unprotected_sentences > 0:
        recommended_prob = target_fraction * n_sentences_total / n_unprotected_sentences

    calibration = {
        'sample_size':              total,
        'target_percent':           args.target_percent,
        'total_docs':               total,
        'protected_docs':           n_protected_docs,
        'total_sentences':          n_sentences_total,
        'protected_sentences':      n_protected_sentences,
        'unprotected_sentences':    n_unprotected_sentences,
        'recommended_removal_prob': recommended_prob,
        'seed':                     args.seed,
        'by_entity':                dict(entity_counts.most_common(50)),
    }
    if apply_removal:
        calibration['applied_removal_prob'] = args.removal_prob
        calibration['sample_kept_docs'] = kept
        calibration['sample_removed_sentences'] = n_removed_sentences
        calibration['sample_realized_percent'] = (
            100.0 * n_removed_sentences / n_sentences_total if n_sentences_total else 0.0
        )

    calib_path = out_dir / 'calibration.json'
    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)

    sep = '-' * 60
    print(f"\n{sep}")
    print(f"  Sampled documents        : {total:,}")
    print(f"  Protected documents      : {n_protected_docs:,} "
          f"({100.0 * n_protected_docs / total:.3f}% of docs)" if total else "")
    print(f"  Total sentences          : {n_sentences_total:,}")
    print(f"  Protected sentences      : {n_protected_sentences:,}")
    print(f"  Unprotected sentences    : {n_unprotected_sentences:,}")
    print(f"  Target removal %         : {args.target_percent}")
    if recommended_prob is None:
        print("  Recommended removal-prob : N/A (no unprotected sentences in sample)")
    elif recommended_prob > 1.0:
        print(f"  Recommended removal-prob : {recommended_prob:.6f}  "
              f"(> 1.0 — target unreachable: the safe/unprotected pool is too "
              f"small relative to --target-percent)")
    else:
        print(f"  Recommended removal-prob : {recommended_prob:.6f}")
    if apply_removal:
        print(f"  Applied removal-prob     : {args.removal_prob}")
        print(f"  Sample realized removal %: {calibration['sample_realized_percent']:.4f}")
        print(f"  Sample filtered output   : {out_dir}/filtered/sample.jsonl")
        print(f"  Sample removed output    : {out_dir}/removed/sample.jsonl")
    print(f"  Calibration written to   : {calib_path}")
    print(sep)


# ---------------------------------------------------------------------------
# Section 8: CLI — Full Mode (subprocess-based parallelism)
# ---------------------------------------------------------------------------

def _load_removal_prob(args: argparse.Namespace) -> float:
    if args.removal_prob is not None:
        return args.removal_prob

    calib_path = Path(args.output) / 'calibration.json'
    if not calib_path.exists():
        print(
            "Error: --removal-prob not given and no calibration.json found at "
            f"{calib_path}. Run --mode subset against the same --output dir "
            "first, or pass --removal-prob explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(calib_path, 'r', encoding='utf-8') as f:
        calib = json.load(f)

    prob = calib.get('recommended_removal_prob')
    if prob is None:
        print(f"Error: {calib_path} has no usable 'recommended_removal_prob'.",
              file=sys.stderr)
        sys.exit(1)
    if prob > 1.0:
        print(f"Error: calibrated removal_prob={prob:.6f} exceeds 1.0 — the "
              "target percentage is not reachable given how many documents "
              "are protected. Lower --target-percent and recalibrate.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Using removal_prob={prob:.6f} from {calib_path} "
          f"(target_percent={calib.get('target_percent')})")
    return prob


def run_full(args: argparse.Namespace) -> None:
    """Download all shards, then process them via independent subprocesses."""
    from huggingface_hub import login

    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("Authenticated with HuggingFace Hub via HF_TOKEN")
    else:
        print("Warning: HF_TOKEN not set. Access may fail for gated datasets.")

    removal_prob = _load_removal_prob(args)

    out_dir   = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / 'cache'

    (out_dir / 'filtered').mkdir(parents=True, exist_ok=True)
    (out_dir / 'removed').mkdir(parents=True,  exist_ok=True)
    (out_dir / 'counts').mkdir(parents=True,   exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Listing Persian shards on HuggingFace Hub ...")
    local_paths = download_all_shards(cache_dir, n_threads=8)
    n_shards = len(local_paths)

    pending = []
    skipped = 0
    for idx, shard_path in enumerate(local_paths):
        fp = out_dir / 'filtered' / f'shard_{idx:04d}.jsonl'
        rp = out_dir / 'removed'  / f'shard_{idx:04d}.jsonl'
        cp = out_dir / 'counts'   / f'shard_{idx:04d}_counts.json'
        if fp.exists() and rp.exists() and cp.exists():
            skipped += 1
            continue
        pending.append((idx, str(shard_path)))

    if skipped:
        print(f"Skipping {skipped} already-completed shards")
    print(f"Processing {len(pending)} shards with {args.workers} workers "
          f"(removal_prob={removal_prob:.6f}) ...")

    n_workers   = max(1, args.workers)
    script_path = str(Path(__file__).resolve())
    pairs_path  = str(Path(args.pairs).resolve())
    output_path = str(out_dir.resolve())

    active: dict[int, tuple] = {}
    completed = 0
    failed    = 0

    while pending or active:
        while len(active) < n_workers and pending:
            idx, shard_path = pending.pop(0)
            cmd = [
                sys.executable, script_path,
                '--worker-mode',
                '--shard', shard_path,
                '--pairs', pairs_path,
                '--output', output_path,
                '--shard-idx', str(idx),
                '--removal-prob', str(removal_prob),
                '--seed', str(args.seed),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[idx] = (proc, time.time())

        for idx in list(active):
            proc, start = active[idx]
            ret = proc.poll()
            if ret is not None:
                output = proc.stdout.read() if proc.stdout else ''
                if output.strip():
                    print(output.strip(), flush=True)
                if ret != 0:
                    print(f"[Shard {idx:04d}] FAILED (exit code {ret})", flush=True)
                    failed += 1
                else:
                    completed += 1
                del active[idx]
            elif time.time() - start > SHARD_TIMEOUT:
                proc.kill()
                proc.wait()
                del active[idx]
                print(f"[Shard {idx:04d}] KILLED (timeout {SHARD_TIMEOUT}s)", flush=True)
                failed += 1

        if active:
            time.sleep(2)

    # Merge per-shard counts
    total_entity    = Counter()
    total_docs               = 0
    total_kept               = 0
    total_protected_docs     = 0
    total_protected_sentences = 0
    total_removed_sentences  = 0
    total_sentences          = 0
    for idx in range(n_shards):
        cp = out_dir / 'counts' / f'shard_{idx:04d}_counts.json'
        if not cp.exists():
            continue
        with open(cp, 'r', encoding='utf-8') as f:
            c = json.load(f)
        total_docs                += c.get('n_rows', 0)
        total_kept                += c.get('n_kept', 0)
        total_protected_docs      += c.get('n_protected_docs', 0)
        total_protected_sentences += c.get('n_protected_sentences', 0)
        total_removed_sentences   += c.get('n_removed_sentences', 0)
        total_sentences           += c.get('n_sentences_total', 0)
        for k, v in c.get('by_entity', {}).items():
            total_entity[k] += v

    realized_percent = (
        100.0 * total_removed_sentences / total_sentences if total_sentences else 0.0
    )

    merged = {
        'total_docs':                total_docs,
        'total_kept_docs':           total_kept,
        'total_protected_docs':      total_protected_docs,
        'total_sentences':           total_sentences,
        'total_protected_sentences': total_protected_sentences,
        'total_removed_sentences':   total_removed_sentences,
        'target_percent':            None,
        'removal_prob_used':         removal_prob,
        'realized_removal_percent':  realized_percent,
        'by_entity':                 dict(total_entity.most_common()),
    }
    calib_path = out_dir / 'calibration.json'
    if calib_path.exists():
        with open(calib_path, 'r', encoding='utf-8') as f:
            merged['target_percent'] = json.load(f).get('target_percent')

    with open(out_dir / 'removal_counts.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    sep = '-' * 60
    print(f"\n{sep}")
    print(f"  Shards               : {n_shards}")
    print(f"  Completed / Failed / Skipped : {completed} / {failed} / {skipped}")
    print(f"  Documents processed  : {total_docs:,}")
    print(f"  Documents kept       : {total_kept:,}")
    print(f"  Protected documents  : {total_protected_docs:,}")
    print(f"  Total sentences      : {total_sentences:,}")
    print(f"  Removed sentences    : {total_removed_sentences:,}")
    print(f"  Removal prob used    : {removal_prob:.6f}")
    print(f"  Target percent       : {merged['target_percent']}")
    print(f"  Realized percent     : {realized_percent:.4f}")
    print(f"  Filtered output      : {out_dir}/filtered/")
    print(f"  Removed sentences    : {out_dir}/removed/")
    print(f"  Merged counts        : {out_dir}/removal_counts.json")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Worker mode (internal, launched by orchestrator) ---
    parser.add_argument('--worker-mode', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--shard', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--shard-idx', type=int, help=argparse.SUPPRESS)

    # --- User-facing arguments ---
    parser.add_argument(
        '--pairs', required=True,
        help='Path to triples JSON file with subjects/objects arrays',
    )
    parser.add_argument(
        '--output', required=True,
        help='Root output directory; filtered/ and removed/ sub-dirs are created',
    )
    parser.add_argument(
        '--mode', choices=['subset', 'full'], default='subset',
        help='subset: calibrate removal_prob (and optionally QA a sample); '
             'full: download all shards and process in parallel',
    )
    parser.add_argument(
        '--sample-size', type=int, default=100_000,
        help='Number of documents for subset/calibration mode (default: 100000)',
    )
    parser.add_argument(
        '--target-percent', type=float, default=8.7,
        help='Desired overall sentence-removal percentage, used by subset '
             'mode to compute the recommended removal probability (default: 8.7)',
    )
    parser.add_argument(
        '--removal-prob', type=float, default=None,
        help='Per-sentence removal probability for unprotected documents. '
             'In subset mode, if given, also applies it to the sample for QA. '
             'In full mode, if omitted, loaded from <output>/calibration.json.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='RNG seed for reproducible random sentence removal (default: 42)',
    )
    parser.add_argument(
        '--workers', type=int, default=min(os.cpu_count() or 1, 32),
        help='Parallel worker processes for full mode '
             f'(default: min(cpu_count, 32) = {min(os.cpu_count() or 1, 32)})',
    )
    parser.add_argument(
        '--cache-dir', type=str, default=None,
        help='Directory for caching downloaded Parquet shards '
             '(default: <output>/cache)',
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.worker_mode:
        if not args.shard or args.shard_idx is None or args.removal_prob is None:
            print("Worker mode requires --shard, --shard-idx and --removal-prob",
                  file=sys.stderr)
            os._exit(1)
        run_worker(args.shard, args.pairs, args.output, args.shard_idx,
                   args.removal_prob, args.seed)
        os._exit(0)

    if args.mode == 'subset':
        run_subset(args)
    else:
        run_full(args)


if __name__ == '__main__':
    main()
    os._exit(0)
