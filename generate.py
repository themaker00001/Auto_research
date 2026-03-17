"""
Generate text from a trained autoresearch model.

Usage:
    uv run generate.py "Your prompt here"
    uv run generate.py "Your prompt here" --max-tokens 300 --temperature 0.9
    uv run generate.py "Your prompt here" --temperature 1.2 --top-k 0   # more random
    uv run generate.py "Your prompt here" --temperature 0.5             # more focused
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from train import GPT, GPTConfig
from prepare import Tokenizer

CHECKPOINT_DIR = Path(__file__).parent / "checkpoint"


def load_model():
    config_path = CHECKPOINT_DIR / "config.json"
    weights_path = CHECKPOINT_DIR / "model.npz"

    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_DIR}.\n"
            "Run `uv run train.py` first to train and save a model."
        )

    with open(config_path) as f:
        config_dict = json.load(f)

    config = GPTConfig(**config_dict)
    model = GPT(config)
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())
    return model


def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    bos_id = tokenizer.get_bos_token_id()

    if prompt:
        token_ids = [bos_id] + tokenizer.encode(prompt)
    else:
        token_ids = [bos_id]

    tokens = mx.array(token_ids, dtype=mx.int32)[None, :]  # [1, T]

    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        # Trim context to model's max sequence length
        inp = tokens[:, -model.config.sequence_len:]
        logits = model(inp)  # [1, T, vocab]
        next_logits = logits[0, -1, :].astype(mx.float32)  # [vocab]

        # Temperature scaling
        next_logits = next_logits / max(temperature, 1e-6)

        # Top-k filtering
        if top_k > 0:
            k = min(top_k, next_logits.shape[-1])
            threshold = mx.sort(next_logits)[-k]
            next_logits = mx.where(
                next_logits >= threshold,
                next_logits,
                mx.full(next_logits.shape, float("-inf")),
            )

        # Sample next token
        next_token = mx.random.categorical(next_logits)
        mx.eval(next_token)
        next_id = int(next_token.item())

        tokens = mx.concatenate(
            [tokens, mx.array([[next_id]], dtype=mx.int32)], axis=1
        )

        print(tokenizer.decode([next_id]), end="", flush=True)

    print()  # final newline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("prompt", nargs="?", default="",
                        help="Text prompt to continue (leave empty for free generation)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Number of new tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature: lower = focused, higher = creative (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k sampling pool size, 0 = disabled (default: 40)")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model()
    print(f"Model loaded ({model.config.n_layer} layers, {model.config.n_embd} dim)\n")

    tokenizer = Tokenizer.from_directory()

    print("--- Generated text ---")
    generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
