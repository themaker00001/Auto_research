# Personal Writing Model — autoresearch-mlx

Train a tiny GPT on your own writing using Karpathy's autonomous research agent loop. The agent runs overnight on an M4 Mac, testing ~80–100 architecture experiments automatically to find the best model for your personal writing style.

Based on [`thenamangoyal/autoresearch`](https://github.com/thenamangoyal/autoresearch) (MLX fork with M4 bug fixes).

## How It Works

1. **Collect** your writing (Obsidian notes, Google Docs, Notion exports, manuscripts) into a single `corpus.txt`
2. **Prepare** a custom BPE tokenizer trained on your vocabulary
3. **Launch** the Claude Code agent, which reads `program.md` and autonomously experiments with model architecture, hyperparameters, and training schedules — committing each improvement to git

## Requirements


- Apple Silicon Mac (M4 recommended, 24GB+ RAM)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

## Setup

```bash
git clone https://github.com/thenamangoyal/autoresearch.git
cd autoresearch
uv sync
```

## Corpus Preparation

### 1. Gather your writing

Place all your writing files into the `input/` folder:

| Source | Format | How to export |
|--------|--------|---------------|
| Obsidian | `.md` | Copy vault folder or specific notes |
| Google Docs | `.docx` | Google Takeout → Drive export |
| Notion | `.md` | Settings → Export all → Markdown |
| Manuscripts | `.docx` | Drop files directly |

Aim for **50,000+ words** minimum. More is better.

### 2. Build the corpus

```bash
python collect_my_writing.py --input input --out corpus.txt
```

This reads all `.md` and `.docx` files recursively, strips formatting, and writes clean text with stats:

```
Found 4 file(s) in 'input':
  chapter_1.docx: 476 words
  chapter_2.docx: 937 words
  novel_draft.docx: 30,905 words
  notes.md: 12,340 words

--- Corpus Stats ---
Total words: 44,658
```

### 3. Train the tokenizer

```bash
rm -rf ~/.cache/autoresearch/    # clear any previous cached data
uv run prepare.py
```

This splits the corpus into 90/10 train/val, then trains a BPE tokenizer (vocab size 2048) on your text.

### 4. Verify with a test training run

```bash
uv run train.py
```

If you hit an OOM error, lower `FINAL_EVAL_BATCH_SIZE` to `16` in `train.py`.

## Running the Agent Loop

```bash
claude --dangerously-skip-permissions "Follow program.md"
```

Leave it running overnight. The agent will:

- Start with a baseline config (depth=4, vocab=2048, seq_len=512)
- Make one change at a time, measuring improvement in bits-per-byte (BPB)
- Try variations: byte-level tokenizer, no warmup, linear decay, different depths
- Commit each improvement with the delta: `[+0.3%] depth=4 vocab=2048 baseline`
- Generate `progress.png` showing the improvement curve

Expect **~10–12 experiments per hour**, or **~80–100 overnight**.

## M4-Specific Tips

- Use `WINDOW_PATTERN = "L"` only — skip banded/sliding window attention
- Lower `FINAL_EVAL_BATCH_SIZE` to `16` if eval causes OOM
- The fork includes fixes for eval OOM and `__main__` guard issues on Apple Silicon

## Key Files

| File | Purpose |
|------|---------|
| `collect_my_writing.py` | Gathers `.md` and `.docx` files into `corpus.txt` |
| `prepare.py` | Splits corpus, trains BPE tokenizer (modified from ClimbMix version) |
| `train.py` | Single training run |
| `program.md` | Instructions the agent follows during the experiment loop |
| `analysis.ipynb` | Notebook to review experiment results |
| `progress.png` | Auto-generated chart of BPB improvement over experiments |

## Philosophy

The corpus is small, narrow, and personal — this is intentional. We're not building a general language model. We want the model to **overfit slightly** on your specific voice, style, and vocabulary. The agent's `program.md` is tuned for this: light regularization, small vocab, and architecture choices suited to a narrow domain.

## .gitignore

`corpus.txt` and `input/` are gitignored to keep personal writing out of version control.