"""
Collect personal writing into a single corpus.txt for autoresearch training.

Reads .md and .docx files from the input folder, strips formatting,
and writes clean text to corpus.txt with document separators.

Usage:
    python collect_my_writing.py --input input --out corpus.txt
"""

import argparse
import os
import re
import sys
from pathlib import Path

def read_markdown(filepath):
    """Read a .md file and strip common markdown formatting."""
    text = filepath.read_text(encoding="utf-8", errors="replace")

    # Remove YAML front matter (---...---)
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

    # Remove images ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove links but keep text [text](url) -> text
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove markdown heading markers but keep text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)

    # Remove inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove horizontal rules
    text = re.sub(r"^[\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def read_docx(filepath):
    """Read a .docx file and extract plain text."""
    try:
        from docx import Document
    except ImportError:
        print("ERROR: python-docx is required for .docx files.")
        print("Install it: pip install python-docx")
        sys.exit(1)

    doc = Document(str(filepath))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def collect_files(input_dir):
    """Recursively find all .md and .docx files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    files = []
    for ext in ["*.md", "*.docx"]:
        files.extend(input_path.rglob(ext))

    # Sort for deterministic output
    files.sort(key=lambda f: f.name.lower())
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Collect personal writing into corpus.txt"
    )
    parser.add_argument(
        "--input",
        default="input",
        help="Input directory containing .md and .docx files (default: input)",
    )
    parser.add_argument(
        "--obsidian",
        default=None,
        help="Alias for --input (backward compat)",
    )
    parser.add_argument(
        "--out",
        default="corpus.txt",
        help="Output corpus file (default: corpus.txt)",
    )
    args = parser.parse_args()

    # --obsidian is an alias for --input
    input_dir = args.obsidian if args.obsidian else args.input

    files = collect_files(input_dir)

    if not files:
        print(f"No .md or .docx files found in '{input_dir}'.")
        print("Add your writing there and try again.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) in '{input_dir}':")

    md_count = 0
    docx_count = 0
    all_texts = []

    for f in files:
        ext = f.suffix.lower()
        try:
            if ext == ".md":
                text = read_markdown(f)
                md_count += 1
            elif ext == ".docx":
                text = read_docx(f)
                docx_count += 1
            else:
                continue

            if text and len(text.strip()) > 0:
                word_count = len(text.split())
                print(f"  {f.name}: {word_count:,} words")
                all_texts.append(text.strip())
            else:
                print(f"  {f.name}: (empty, skipped)")

        except Exception as e:
            print(f"  {f.name}: ERROR - {e}")

    if not all_texts:
        print("\nNo text extracted from any files.")
        sys.exit(1)

    # Join with double newlines as document separator
    corpus = "\n\n".join(all_texts)
    total_words = len(corpus.split())
    total_chars = len(corpus)

    # Write output
    out_path = Path(args.out)
    out_path.write_text(corpus, encoding="utf-8")

    print(f"\n--- Corpus Stats ---")
    print(f"Files processed: {md_count} .md, {docx_count} .docx")
    print(f"Total words:     {total_words:,}")
    print(f"Total chars:     {total_chars:,}")
    print(f"Output:          {out_path}")

    if total_words < 50_000:
        print(f"\nWARNING: {total_words:,} words is below the 50k recommended minimum.")
        print("The model will still train, but consider adding more writing:")
        print("  - Notion export (Settings > Export > Markdown)")
        print("  - More Obsidian notes")
        print("  - Blog posts, journal entries, emails")
    else:
        print(f"\nLooks good! {total_words:,} words is enough for training.")


if __name__ == "__main__":
    main()