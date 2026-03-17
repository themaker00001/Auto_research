"""
One-time data preparation for autoresearch experiments (MLX version).
Modified: loads personal writing from corpus.txt instead of ClimbMix shards.

Usage:
    python prepare.py                  # full prep (split corpus + tokenizer)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
import random
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import rustbpe
import tiktoken
import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 3 * 524288 # number of tokens for val eval (reduced for Apple Silicon)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# --- Personal corpus config ---
CORPUS_FILE = Path(__file__).parent / "corpus.txt"
TRAIN_DOCS_PKL = os.path.join(DATA_DIR, "train_docs.pkl")
VAL_DOCS_PKL = os.path.join(DATA_DIR, "val_docs.pkl")
VAL_SPLIT_RATIO = 0.10  # 10% held out for validation

VOCAB_SIZE = 2048  # small vocab for small personal corpus

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Corpus loading and splitting
# ---------------------------------------------------------------------------

def load_and_split_corpus():
    """Read corpus.txt, split into documents, save train/val splits."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_DOCS_PKL) and os.path.exists(VAL_DOCS_PKL):
        print(f"Corpus: train/val splits already exist at {DATA_DIR}")
        with open(TRAIN_DOCS_PKL, "rb") as f:
            train_docs = pickle.load(f)
        with open(VAL_DOCS_PKL, "rb") as f:
            val_docs = pickle.load(f)
        print(f"  Train: {len(train_docs)} docs, Val: {len(val_docs)} docs")
        return train_docs, val_docs

    if not CORPUS_FILE.exists():
        print(f"ERROR: {CORPUS_FILE} not found.")
        print("Run collect_my_writing.py first to generate it.")
        sys.exit(1)

    raw_text = CORPUS_FILE.read_text(encoding="utf-8")
    print(f"Corpus: loaded {len(raw_text):,} chars ({len(raw_text.split()):,} words)")

    # Split into documents on double newlines (paragraphs/sections)
    docs = [d.strip() for d in raw_text.split("\n\n") if d.strip()]

    # Filter out very short fragments (< 20 chars)
    docs = [d for d in docs if len(d) >= 20]
    print(f"Corpus: {len(docs)} documents after splitting")

    if len(docs) < 10:
        print("WARNING: Very few documents. Consider adding more writing to input_docs/.")

    # Shuffle and split
    random.seed(42)
    random.shuffle(docs)
    val_count = max(1, int(len(docs) * VAL_SPLIT_RATIO))
    val_docs = docs[:val_count]
    train_docs = docs[val_count:]

    train_words = sum(len(d.split()) for d in train_docs)
    val_words = sum(len(d.split()) for d in val_docs)
    print(f"  Train: {len(train_docs)} docs ({train_words:,} words)")
    print(f"  Val:   {len(val_docs)} docs ({val_words:,} words)")

    with open(TRAIN_DOCS_PKL, "wb") as f:
        pickle.dump(train_docs, f)
    with open(VAL_DOCS_PKL, "wb") as f:
        pickle.dump(val_docs, f)

    return train_docs, val_docs

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def text_iterator(max_chars=1_000_000_000):
    """Yield documents from training split for tokenizer training."""
    with open(TRAIN_DOCS_PKL, "rb") as f:
        train_docs = pickle.load(f)
    nchars = 0
    for doc in train_docs:
        nchars += len(doc)
        yield doc
        if nchars >= max_chars:
            return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_DOCS_PKL):
        print("Tokenizer: train split not found. Run corpus splitting first.")
        sys.exit(1)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # Build token_bytes lookup for BPB evaluation
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_arr = np.array(token_bytes_list, dtype=np.int32)
    np.save(token_bytes_path, token_bytes_arr)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes():
    """Load token_bytes lookup as an mx.array."""
    path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing token_bytes lookup at {path}. Run prepare.py first.")
    return mx.array(np.load(path), dtype=mx.int32)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from pickled splits."""
    if split == "train":
        docs_path = TRAIN_DOCS_PKL
    else:
        docs_path = VAL_DOCS_PKL

    with open(docs_path, "rb") as f:
        all_docs = pickle.load(f)

    assert len(all_docs) > 0, f"No documents found in {split} split."

    epoch = 1
    while True:
        for i in range(0, len(all_docs), tokenizer_batch_size):
            yield all_docs[i:i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).

    Returns mx.array batches directly (unified memory — no CPU/GPU transfer needed).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    while True:
        all_rows = []
        for _ in range(B):
            row = []
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
                    pos += remaining

            all_rows.append(row[:row_capacity])

        # Convert to mx.array — unified memory, no transfers needed
        row_array = mx.array(all_rows, dtype=mx.int32)
        inputs = row_array[:, :-1]
        targets = row_array[:, 1:]
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes()
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = mx.take(token_bytes, y_flat, axis=0)
        mask = nbytes > 0
        total_nats += mx.sum(loss_flat * mask).item()
        total_bytes += int(mx.sum(nbytes).item())

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch (MLX) — personal corpus mode")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Corpus file: {CORPUS_FILE}")
    print()

    # Step 1: Load corpus and split into train/val
    load_and_split_corpus()
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")