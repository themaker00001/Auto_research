# Personal Writing Model — Research Program

## Goal
Minimize `val_bpb` on a personal English writing corpus (~1–10MB of notes,
docs, and long-form writing). The corpus is narrow, stylistically consistent,
and much smaller than web-scale data. This changes what good architecture
looks like significantly.

## What success looks like
- val_bpb drops below 1.5 within the first few runs
- The model, when sampled, produces text that sounds like the author wrote it
- Improvements transfer: a change that helps on the first 50% of corpus
  should also help on the held-out 50%

## Dataset
- File: `corpus.txt` in the repo root
- Format: plain English text, documents separated by `\n\n---\n\n`
- Language: English only
- Characteristics: personal voice, consistent style, varied topics
  (notes, essays, docs), medium sentence length

## Key differences from the default ClimbMix setup
The corpus is much smaller and narrower than ClimbMix. This means:
1. The model can and should overfit somewhat — we *want* it to learn this
   specific style, not generalize to all English
2. Smaller architectures likely win — don't need depth to compress a small corpus
3. Tokenization matters more — the vocabulary should fit this specific writing style
4. Regularization should be light — weight decay that helps on web text may hurt here

## Suggested experiments (in rough priority order)

### Tokenizer / vocabulary
- Try smaller vocab: 1024 or 2048 instead of default 8192
  (personal corpus doesn't need a large vocabulary)
- Try byte-level tokenizer (vocab=256): completely vocab-free,
  no out-of-vocabulary issue, often strong on narrow corpora
- Try 4096 as a middle ground

### Architecture — start here
- Reduce DEPTH from 8 → 4 or even 3 (corpus is small, shallow is fine)
- Try wider at same depth: more heads, larger d_model at DEPTH=3
- Reduce MAX_SEQ_LEN to 512 (personal writing rarely has long dependencies)
- Experiment with no sliding window attention (WINDOW_PATTERN = "L" only)
  since we don't need long-range context modeling

### Training dynamics
- Try higher learning rate (corpus is small, we can move faster)
- Try WARMUP_RATIO = 0.0 (no warmup — 5 min budget is short)
- Linear weight decay schedule: decay to 0 by end of training
  (prevents over-regularization during warmdown)
- Try larger DEVICE_BATCH_SIZE since sequences are shorter

### Sequence length
- 256 or 512 is likely sufficient for personal writing style
- Shorter sequences = more steps per 5 min = better optimization
- Don't go below 128 — style needs some context to be meaningful

## What NOT to change
- `prepare.py` — do not touch
- The corpus file path (already configured)
- The 5-minute training budget
- The val_bpb metric definition

## Experiment strategy
Make one change at a time. The corpus is small enough that a single run
gives a reliable signal. If val_bpb improves by >0.5%, keep the change.
If it's ambiguous (within 0.2%), try it again with a different random seed
before committing.

Start with DEPTH=4, vocab=2048, MAX_SEQ_LEN=512. This is the baseline
hypothesis for a small personal corpus. Everything else is measured
relative to this starting point.

## Commit message format
Use descriptive commits: `[+0.3%] depth=4 vocab=2048 seq=512 baseline`
or `[-1.2%] reverted: byte-level tokenizer worse than bpe-2048`

This makes the git log readable as a research diary.