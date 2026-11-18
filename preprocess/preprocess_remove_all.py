#!/usr/bin/env python3
"""
preprocess_remove_all.py

Process sentences containing any known subject from the FineWeb2 Persian
dataset, regardless of whether an object also appears.  Two sentence-level
modes are supported (--sentence-mode):

  remove  (default) : drop matching sentences entirely
  replace           : swap matching sentences with a random Persian template
                      where {subject} is filled with the matched subject

Outputs two JSONL directories:

  - filtered/  : documents with matching sentences removed or replaced
  - removed/   : extracted sentences with metadata (doc_id, subject, relation,
                 and — in replace mode — the replacement sentence)

Parallelism uses independent subprocesses (no fork) to avoid PyArrow/datasets
crashes.  The same file serves as both orchestrator (default) and per-shard
worker (--worker-mode).

Usage
-----
  # 1. Quick test — remove mode:
  python preprocess_remove_all.py \\
      --pairs         data/facts/triples-en-fa.json \\
      --output        data/removed_facts_dataset_sent_all \\
      --mode          subset \\
      --sentence-mode remove \\
      --sample-size   1000

  # 2. Quick test — replace mode:
  python preprocess_remove_all.py \\
      --pairs         data/facts/triples-en-fa.json \\
      --output        data/replaced_facts_dataset_sent_all \\
      --mode          subset \\
      --sentence-mode replace \\
      --sample-size   1000

  # 3. Full parallel run:
  HF_TOKEN=hf_xxx python preprocess_remove_all.py \\
      --pairs         data/facts/triples-en-fa.json \\
      --output        data/removed_facts_dataset_sent_all \\
      --mode          full \\
      --sentence-mode remove \\
      --workers       16 \\
      --cache-dir     data/removed_facts_dataset_sent_all/cache
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# ParsiNorm import (local copy)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "ParsiNorm"))
from parsinorm import Tokenizer as _ParsiNormTokenizer

# ---------------------------------------------------------------------------
# Lazy imports — only loaded when actually needed (workers need pyarrow,
# orchestrator needs huggingface_hub, subset mode needs datasets).
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


_romanizer = None


def _get_romanizer():
    global _romanizer
    if _romanizer is None:
        import uroman as ur
        _romanizer = ur.Uroman()
    return _romanizer


def transliterate_sentence(text: str) -> str:
    """Return the uroman Latin transliteration of *text*."""
    return _get_romanizer().romanize_string(text)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_REPO  = "HuggingFaceFW/fineweb-2"
LANGUAGE_CODE = "fas_Arab"
PROXIMITY_WINDOW = 300   # max chars for a sentence before chunking
SHARD_TIMEOUT = 2 * 3600 # 2 hours per shard

def load_relation_templates(json_path: str) -> tuple[dict, list]:
    """Load per-relation sentence templates from a JSON file.

    Returns (templates_by_relation, flat_fallback_list).
    The flat list is used when a relation is not found in the dict.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    flat = [t for lst in templates.values() for t in lst]
    return templates, flat

# ---------------------------------------------------------------------------
# Section 1: Normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(
    r'[،؛؟۔«»(){}\[\]/\\"\'`|^~*+=#@$%&'   # Persian / Arabic punctuation
    r',;?!.()\[\]{}/:–—\-'                    # ASCII punctuation
    r'\u2018\u2019\u201c\u201d\u2039\u203a]'  # typographic quotes
)


def normalize_for_matching(text: str) -> str:
    """Normalise text for matching purposes only -- never written to output."""
    text = text.replace('\u200c', ' ')   # ZWNJ / half-space
    text = text.replace('\u200b', ' ')   # zero-width space
    text = text.replace('\u00a0', ' ')   # non-breaking space
    text = _PUNCT_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Section 2: Subject Matching (Aho-Corasick)
# ---------------------------------------------------------------------------

try:
    import ahocorasick
    _HAS_AHOCORASICK = True
except ImportError:
    _HAS_AHOCORASICK = False


class SubjectMatcher:
    """Load triples JSON, build an Aho-Corasick automaton for fast subject matching.

    A sentence is removed if any known subject appears in it, regardless of
    whether an object is also present.
    """

    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # entries[i] = {subjects, norm_subjects, relation, subject_en, subject}
        self.entries = []
        # norm_subject_term -> list of entry_idx
        self._term_to_entries: dict[str, list[int]] = {}

        all_terms: set[str] = set()

        for item in data:
            subjects = item.get('subjects', [])
            if not subjects:
                continue

            norm_subjects = list({normalize_for_matching(s) for s in subjects if s.strip()})
            if not norm_subjects:
                continue

            entry_idx = len(self.entries)
            self.entries.append({
                'subjects':      subjects,
                'norm_subjects': norm_subjects,
                'relation':      item.get('relation', ''),
                'subject_en':    item.get('subject_en', ''),
                'subject':       item.get('subject', subjects[0]),
            })

            for ns in norm_subjects:
                all_terms.add(ns)
                self._term_to_entries.setdefault(ns, []).append(entry_idx)

        # Build Aho-Corasick automaton over subject terms only
        if _HAS_AHOCORASICK and all_terms:
            self._automaton = ahocorasick.Automaton()
            for term in all_terms:
                self._automaton.add_word(term, term)
            self._automaton.make_automaton()
            self._use_ac = True
        else:
            self._use_ac = False
            self._all_norm_subjects = sorted(all_terms, key=len, reverse=True)

    def find_match(self, sentence: str):
        """Return (True, subject, relation) or (False, None, None).

        *sentence* is the original (un-normalised) text.  Normalisation is
        applied internally for matching only.  Only the presence of a subject
        is required — no object check is performed.
        """
        norm = ' ' + normalize_for_matching(sentence) + ' '

        if self._use_ac:
            return self._find_match_ac(norm)
        else:
            return self._find_match_linear(norm)

    def _find_match_ac(self, norm: str):
        """Aho-Corasick single-pass matching."""
        for end_idx, term in self._automaton.iter(norm):
            start_idx = end_idx - len(term) + 1
            # Word-boundary check: character before start and after end must be space
            if norm[start_idx - 1] == ' ' and (end_idx + 1 >= len(norm) or norm[end_idx + 1] == ' '):
                # Found a subject term — return the first match
                entry_idx = self._term_to_entries[term][0]
                entry = self.entries[entry_idx]
                subj_orig = self._find_original_subject(entry, term)
                return True, subj_orig, entry['relation']

        return False, None, None

    def _find_match_linear(self, norm: str):
        """Fallback linear scan (no pyahocorasick installed)."""
        for entry in self.entries:
            for subj, ns in zip(entry['subjects'], entry['norm_subjects']):
                if ' ' + ns + ' ' in norm:
                    return True, subj, entry['relation']

        return False, None, None

    @staticmethod
    def _find_original_subject(entry, norm_term):
        for s in entry['subjects']:
            if normalize_for_matching(s) == norm_term:
                return s
        return entry['subject']


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
        # Fallback: return the whole line as one sentence
        return [text]


def chunk_sentence(sentence: str, max_chars: int = PROXIMITY_WINDOW) -> list[str]:
    """Split a long sentence into chunks at word boundaries.

    Each chunk is <= *max_chars* characters.  We split at the last space
    before the limit so words stay intact.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    chunks = []
    remaining = sentence
    while len(remaining) > max_chars:
        # Find last space within the limit
        split_at = remaining.rfind(' ', 0, max_chars)
        if split_at <= 0:
            # No space found — force split at max_chars
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


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

def process_document(row: dict, matcher: SubjectMatcher,
                     sentence_mode: str = 'remove',
                     templates_by_relation: dict | None = None,
                     templates_flat: list | None = None,
                     transliterate_mode: str | None = None) -> tuple:
    """Filter one FineWeb2 document at sentence granularity.

    1. Split text on ``\\n`` to preserve paragraph structure.
    2. For each line, split into sentences via ParsiNorm.
    3. Chunk sentences > 300 chars at word boundaries.
    4. For each chunk:
       - No match      : keep it.
       - remove match  : drop it entirely.
       - replace match : substitute with a relation-specific template.
    5. Rejoin kept sentences within each line with space, lines with ``\\n``.
    6. Transliteration (uroman):
       - 'sentence' : append each sentence's transliteration immediately
                      after the sentence in the same pass (O(n)).
       - 'document' : transliterate the entire filtered document once and
                      append it after two newlines at the end.
       - None       : no transliteration.

    Returns (filtered_text | None, removed_list).
    In replace mode, removed_list entries include a ``replacement`` field.
    """
    text = row.get('text', '') or ''   # guard against None in parquet null fields
    if not text.strip():
        return None, []

    doc_id = row.get('id',  '')
    url    = row.get('url', '')

    lines        = text.split('\n')
    kept_lines   = []
    removed_list = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append('')
            continue

        sentences = split_sentences(stripped)
        kept_sents = []

        for sent in sentences:
            chunks = chunk_sentence(sent) if len(sent) > PROXIMITY_WINDOW else [sent]

            for chunk in chunks:
                matched, subj, rel = matcher.find_match(chunk)
                if matched:
                    record = {
                        'doc_id':   doc_id,
                        'url':      url,
                        'sentence': chunk,
                        'subject':  subj,
                        'relation': rel,
                    }
                    if sentence_mode == 'replace':
                        template_list = (
                            templates_by_relation.get(rel)
                            if templates_by_relation and rel
                            else None
                        ) or templates_flat or []
                        template    = random.choice(template_list)
                        replacement = template.format(subject=subj)
                        record['replacement'] = replacement
                        kept_sents.append(replacement)
                        if transliterate_mode == 'sentence':
                            kept_sents.append(transliterate_sentence(replacement))
                    # remove mode: chunk is simply not added to kept_sents
                    removed_list.append(record)
                else:
                    kept_sents.append(chunk)
                    if transliterate_mode == 'sentence':
                        kept_sents.append(transliterate_sentence(chunk))

        if kept_sents:
            kept_lines.append(' '.join(kept_sents))

    # Trim trailing empty lines
    while kept_lines and not kept_lines[-1].strip():
        kept_lines.pop()

    filtered_text = '\n'.join(kept_lines).strip()
    if filtered_text and transliterate_mode == 'document':
        filtered_text = filtered_text + '\n\n' + transliterate_sentence(filtered_text)
    return (filtered_text if filtered_text else None), removed_list


# ---------------------------------------------------------------------------
# Section 6: Shard Worker
# ---------------------------------------------------------------------------

def run_worker(shard_path: str, pairs_path: str, output_dir: str,
               shard_idx: int, sentence_mode: str = 'remove',
               templates_path: str | None = None,
               transliterate_mode: str | None = None) -> None:
    """Process one Parquet shard end-to-end.  Called as a subprocess."""
    out = Path(output_dir)
    filtered_path = out / 'filtered' / f'shard_{shard_idx:04d}.jsonl'
    removed_path  = out / 'removed'  / f'shard_{shard_idx:04d}.jsonl'
    counts_path   = out / 'counts'   / f'shard_{shard_idx:04d}_counts.json'

    # Resume support
    if filtered_path.exists() and removed_path.exists() and counts_path.exists():
        print(f"[Shard {shard_idx:04d}] SKIP (already done)", flush=True)
        return

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)

    matcher = SubjectMatcher(pairs_path)
    print(f"[Shard {shard_idx:04d}] Loaded {len(matcher.entries)} entries, "
          f"AC={'yes' if matcher._use_ac else 'no'}", flush=True)

    tmpl_by_rel, tmpl_flat = None, None
    if sentence_mode == 'replace' and templates_path:
        tmpl_by_rel, tmpl_flat = load_relation_templates(templates_path)

    # Warm up romanizer once per worker process if needed
    if transliterate_mode:
        _get_romanizer()

    n_rows = n_kept = n_removed = 0
    subj_counts = Counter()

    with open(filtered_path, 'w', encoding='utf-8') as ff, \
         open(removed_path,  'w', encoding='utf-8') as rf:

        for row in iter_parquet_rows(shard_path):
            n_rows += 1
            ftext, removed = process_document(
                row, matcher, sentence_mode, tmpl_by_rel, tmpl_flat,
                transliterate_mode)

            if ftext is not None:
                row['text'] = ftext
                ff.write(json.dumps(row, ensure_ascii=False) + '\n')
                n_kept += 1

            for r in removed:
                rf.write(json.dumps(r, ensure_ascii=False) + '\n')
                subj_counts[r['subject']] += 1
            n_removed += len(removed)

    # Save per-shard counts (include row totals so orchestrator can aggregate them)
    counts = {
        'n_rows':     n_rows,
        'n_kept':     n_kept,
        'n_removed':  n_removed,
        'by_subject': dict(subj_counts.most_common()),
    }
    with open(counts_path, 'w', encoding='utf-8') as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    print(f"[Shard {shard_idx:04d}] DONE  rows={n_rows:,}  "
          f"kept={n_kept:,}  removed_sents={n_removed:,}", flush=True)


# ---------------------------------------------------------------------------
# Section 7: CLI — Subset Mode
# ---------------------------------------------------------------------------

def run_subset(args: argparse.Namespace) -> None:
    """Stream first N documents, process sequentially, write output."""
    from datasets import load_dataset
    from tqdm import tqdm

    matcher = SubjectMatcher(args.pairs)
    print(f"Loaded {len(matcher.entries)} entries from {args.pairs}")
    print(f"Aho-Corasick: {'enabled' if matcher._use_ac else 'disabled (install pyahocorasick)'}")
    print(f"Sentence mode: {args.sentence_mode}")

    tmpl_by_rel, tmpl_flat = None, None
    if args.sentence_mode == 'replace':
        if not args.templates:
            print("Error: --templates is required when --sentence-mode replace", file=sys.stderr)
            sys.exit(1)
        tmpl_by_rel, tmpl_flat = load_relation_templates(args.templates)
        print(f"Loaded templates for {len(tmpl_by_rel)} relations from {args.templates}")

    if args.transliterate:
        _get_romanizer()   # warm up once before the document loop
        print(f"Transliteration: {args.transliterate}")

    out_dir = Path(args.output)
    (out_dir / 'filtered').mkdir(parents=True, exist_ok=True)
    (out_dir / 'removed').mkdir(parents=True,  exist_ok=True)

    filtered_path = out_dir / 'filtered' / 'sample.jsonl'
    removed_path  = out_dir / 'removed'  / 'sample.jsonl'

    print(f"Streaming first {args.sample_size} documents "
          f"from {DATASET_REPO} (lang={LANGUAGE_CODE}) ...")

    dataset = load_dataset(
        DATASET_REPO,
        name=LANGUAGE_CODE,
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    total = kept = discarded = removed_sents = 0
    subj_counts = Counter()

    with open(filtered_path, 'w', encoding='utf-8') as ff, \
         open(removed_path,  'w', encoding='utf-8') as rf, \
         tqdm(total=args.sample_size, desc="Documents",
              unit="doc", dynamic_ncols=True) as pbar:

        for record in dataset:
            if total >= args.sample_size:
                break

            ftext, removed = process_document(
                record, matcher, args.sentence_mode, tmpl_by_rel, tmpl_flat,
                args.transliterate)

            if ftext is not None:
                record['text'] = ftext
                ff.write(json.dumps(record, ensure_ascii=False) + '\n')
                kept += 1
            else:
                discarded += 1

            for r in removed:
                rf.write(json.dumps(r, ensure_ascii=False) + '\n')
                subj_counts[r['subject']] += 1

            removed_sents += len(removed)
            total += 1
            pbar.update(1)
            pbar.set_postfix(kept=kept, discarded=discarded,
                             removed_sents=removed_sents)

    # Save counts
    counts = {
        'by_subject': dict(subj_counts.most_common()),
    }
    counts_path = out_dir / 'sample_match_counts.json'
    with open(counts_path, 'w', encoding='utf-8') as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    _print_summary(mode="subset", n_shards=1, total_docs=total,
                   total_kept=kept, total_removed_sents=removed_sents,
                   out_dir=out_dir)


# ---------------------------------------------------------------------------
# Section 8: CLI — Full Mode (subprocess-based parallelism)
# ---------------------------------------------------------------------------

def run_full(args: argparse.Namespace) -> None:
    """Download all shards, then process them via independent subprocesses."""
    from huggingface_hub import login

    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("Authenticated with HuggingFace Hub via HF_TOKEN")
    else:
        print("Warning: HF_TOKEN not set. Access may fail for gated datasets.")

    out_dir   = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / 'cache'

    (out_dir / 'filtered').mkdir(parents=True, exist_ok=True)
    (out_dir / 'removed').mkdir(parents=True,  exist_ok=True)
    (out_dir / 'counts').mkdir(parents=True,   exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Listing Persian shards on HuggingFace Hub ...")
    local_paths = download_all_shards(cache_dir, n_threads=8)
    n_shards = len(local_paths)

    # Build list of shards to process (skip already-done ones)
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
    print(f"Processing {len(pending)} shards with {args.workers} workers ...")

    n_workers = max(1, args.workers)
    script_path = str(Path(__file__).resolve())
    pairs_path  = str(Path(args.pairs).resolve())
    output_path = str(out_dir.resolve())

    active: dict[int, tuple] = {}   # shard_idx -> (Popen, start_time)
    completed = 0
    failed    = 0

    while pending or active:
        # Fill worker slots
        while len(active) < n_workers and pending:
            idx, shard_path = pending.pop(0)
            cmd = [
                sys.executable, script_path,
                '--worker-mode',
                '--shard', shard_path,
                '--pairs', pairs_path,
                '--output', output_path,
                '--shard-idx', str(idx),
                '--sentence-mode', args.sentence_mode,
            ]
            if args.sentence_mode == 'replace' and args.templates:
                cmd += ['--templates', str(Path(args.templates).resolve())]
            if args.transliterate:
                cmd += ['--transliterate', args.transliterate]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[idx] = (proc, time.time())

        # Check for completed / timed-out workers
        for idx in list(active):
            proc, start = active[idx]
            ret = proc.poll()
            if ret is not None:
                # Worker finished
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
    total_subj    = Counter()
    total_docs    = 0
    total_kept    = 0
    total_removed = 0
    for idx in range(n_shards):
        cp = out_dir / 'counts' / f'shard_{idx:04d}_counts.json'
        if not cp.exists():
            continue
        with open(cp, 'r', encoding='utf-8') as f:
            c = json.load(f)
        total_docs    += c.get('n_rows',    0)
        total_kept    += c.get('n_kept',    0)
        total_removed += c.get('n_removed', 0)
        for k, v in c.get('by_subject', {}).items():
            total_subj[k] += v

    merged = {
        'by_subject': dict(total_subj.most_common()),
    }
    with open(out_dir / 'match_counts.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    _print_summary(
        mode="full",
        n_shards=n_shards,
        total_docs=total_docs,
        total_kept=total_kept,
        total_removed_sents=total_removed,
        out_dir=out_dir,
        extra=f"  Completed shards    : {completed}\n"
              f"  Failed shards       : {failed}\n"
              f"  Skipped shards      : {skipped}",
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(*, mode, n_shards, total_docs, total_kept,
                   total_removed_sents, out_dir, extra=''):
    sep = '-' * 52
    print(f"\n{sep}")
    print(f"  Mode                : {mode}")
    print(f"  Shards              : {n_shards}")
    if total_docs:
        print(f"  Documents processed : {total_docs:,}")
        print(f"  Documents kept      : {total_kept:,}")
        print(f"  Documents discarded : {total_docs - total_kept:,}")
    print(f"  Sentences removed   : {total_removed_sents:,}")
    if extra:
        print(extra)
    print(f"  Filtered output     : {out_dir}/filtered/")
    print(f"  Removed sentences   : {out_dir}/removed/")
    print(f"  Match counts        : {out_dir}/match_counts.json")
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
        help='Path to triples JSON file with subjects arrays',
    )
    parser.add_argument(
        '--output', required=True,
        help='Root output directory; filtered/ and removed/ sub-dirs are created',
    )
    parser.add_argument(
        '--mode', choices=['subset', 'full'], default='subset',
        help='subset: stream first N docs for testing (default); '
             'full: download all shards and process in parallel',
    )
    parser.add_argument(
        '--sample-size', type=int, default=1000,
        help='Number of documents for subset mode (default: 1000)',
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
    parser.add_argument(
        '--sentence-mode', choices=['remove', 'replace'], default='remove',
        help='remove: drop matching sentences (default); '
             'replace: substitute with a random relation-specific template',
    )
    parser.add_argument(
        '--templates', type=str, default=None,
        help='Path to per-relation sentence templates JSON '
             '(required when --sentence-mode replace)',
    )
    parser.add_argument(
        '--transliterate', choices=['sentence', 'document'], default=None,
        help='sentence: append uroman transliteration after each sentence; '
             'document: append transliteration of the whole filtered document '
             'at the end; omit to disable transliteration',
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    # --- Worker mode: process a single shard and exit ---
    if args.worker_mode:
        if not args.shard or args.shard_idx is None:
            print("Worker mode requires --shard and --shard-idx", file=sys.stderr)
            os._exit(1)
        run_worker(args.shard, args.pairs, args.output, args.shard_idx,
                   args.sentence_mode, args.templates, args.transliterate or None)
        # Use os._exit instead of sys.exit to bypass Python's shutdown sequence.
        # sys.exit triggers PyArrow's C++ thread destructors which crash with
        # SIGABRT (exit 134) — making the orchestrator mark the shard as FAILED
        # even though all output was written correctly.
        os._exit(0)

    # --- Orchestrator mode ---
    if args.mode == 'subset':
        run_subset(args)
    else:
        run_full(args)


if __name__ == '__main__':
    main()
    # PyArrow background threads outlive the GIL during Python's normal shutdown,
    # causing a spurious "terminate called without an active exception" abort.
    # os._exit() skips the finalizer entirely — all file writes are complete.
    os._exit(0)
