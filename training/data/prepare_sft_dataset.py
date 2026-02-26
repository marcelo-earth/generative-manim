"""Merge all data sources, validate, normalize format, split into train/val/test."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.code_extraction import normalize_class_name, ensure_manim_import, clean_code
from utils.system_prompt import SYSTEM_PROMPT
from rendering.manim_verifier import verify_code

BASE_DIR = Path(__file__).parent.parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
EXTRA_DATASET = BASE_DIR / "dataset-2025-02-15T02-45-51-960Z.jsonl"


def load_messages_dataset(path: Path, source: str) -> List[Dict]:
    """Load a JSONL dataset in messages format (edoh, physics, feb2025)."""
    examples = []
    if not path.exists():
        print(f"  [skip] {path} not found")
        return examples
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            msgs = data.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if user and assistant:
                examples.append({"prompt": user, "code": assistant, "source": source})
    print(f"  Loaded {len(examples)} from {path.name}")
    return examples


def load_claude_generated(path: Path) -> List[Dict]:
    """Load claude-generated prompt/code pairs."""
    examples = []
    if not path.exists():
        print(f"  [skip] {path} not found")
        return examples
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "prompt": data["prompt"],
                "code": data["code"],
                "source": "claude_generated",
            })
    print(f"  Loaded {len(examples)} from {path.name}")
    return examples


def load_code_samples(code_dir: Path) -> List[Dict]:
    """Load standalone .py code samples (no prompts)."""
    examples = []
    if not code_dir.exists():
        return examples
    for py_file in sorted(code_dir.glob("*.py")):
        code = py_file.read_text()
        examples.append({
            "prompt": None,
            "code": code,
            "source": f"code/{py_file.name}",
        })
    print(f"  Loaded {len(examples)} code samples")
    return examples


def normalize_example(example: Dict) -> Optional[Dict]:
    """Normalize a single example into chat messages format."""
    if not example.get("prompt"):
        return None  # Skip code-only samples (no prompt for SFT)

    code = clean_code(example["code"])

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": code},
        ],
        "source": example.get("source", "unknown"),
    }


def validate_examples(
    examples: List[Dict],
    timeout: int = 60,
    skip_validation: bool = False,
) -> List[Dict]:
    """Filter examples to only those that render successfully."""
    if skip_validation:
        print(f"  Skipping validation, keeping all {len(examples)} examples")
        return examples

    valid = []
    fail = 0
    for i, ex in enumerate(examples):
        code = ex["messages"][2]["content"]  # assistant message
        result = verify_code(code, timeout=timeout)
        if result.success:
            valid.append(ex)
        else:
            fail += 1
        if (i + 1) % 25 == 0:
            print(f"    Validated {i+1}/{len(examples)} -- {len(valid)} OK, {fail} failed")

    print(f"  Validation complete: {len(valid)}/{len(examples)} render OK ({len(valid)/len(examples)*100:.1f}%)")
    return valid


def prepare_dataset(
    output_dir: Path = OUTPUTS_DIR,
    test_size: int = 50,
    val_ratio: float = 0.1,
    seed: int = 42,
    skip_validation: bool = False,
    timeout: int = 60,
):
    """Full pipeline: load all sources, normalize, validate, split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")

    all_raw = []

    # 1. Claude-generated (our highest quality, pre-validated)
    all_raw.extend(load_claude_generated(OUTPUTS_DIR / "claude_generated.jsonl"))

    # 2. Edoh dataset (PEP8 formatted version with proper newlines)
    all_raw.extend(load_messages_dataset(
        DATASETS_DIR / "edoh-dataset-format-pep8.jsonl", "edoh"
    ))

    # 3. Physics dataset
    all_raw.extend(load_messages_dataset(
        DATASETS_DIR / "physics-01.jsonl", "physics"
    ))

    # 4. Feb 2025 export
    all_raw.extend(load_messages_dataset(EXTRA_DATASET, "feb2025"))

    # 5. Code samples (standalone .py files, no prompts -- skip for SFT)
    # code_samples = load_code_samples(DATASETS_DIR / "code")
    # all_raw.extend(code_samples)

    print(f"\nTotal raw examples: {len(all_raw)}")
    by_source = Counter(e["source"] for e in all_raw)
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")

    # Normalize all examples
    print("\nNormalizing...")
    normalized = []
    skipped = 0
    for ex in all_raw:
        result = normalize_example(ex)
        if result:
            normalized.append(result)
        else:
            skipped += 1
    print(f"  Normalized: {len(normalized)} (skipped {skipped} with no prompt)")

    # Deduplicate by prompt text
    seen = set()
    deduped = []
    dupes = 0
    for ex in normalized:
        prompt = ex["messages"][1]["content"].strip().lower()
        if prompt not in seen:
            seen.add(prompt)
            deduped.append(ex)
        else:
            dupes += 1
    print(f"  After dedup: {len(deduped)} (removed {dupes} duplicates)")

    # Validate (render each example)
    print(f"\nValidating renders (timeout={timeout}s)...")
    valid = validate_examples(deduped, timeout=timeout, skip_validation=skip_validation)

    # Shuffle and split
    random.seed(seed)
    random.shuffle(valid)

    # Adjust test_size if dataset is small
    actual_test_size = min(test_size, len(valid) // 5)  # at most 20% for test
    test_set = valid[:actual_test_size]
    remaining = valid[actual_test_size:]
    val_size = max(1, int(len(remaining) * val_ratio))
    val_set = remaining[:val_size]
    train_set = remaining[val_size:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # Source breakdown per split
    for name, data in [("Train", train_set), ("Val", val_set), ("Test", test_set)]:
        sources = Counter(ex.get("source", "unknown") for ex in data)
        breakdown = ", ".join(f"{s}={c}" for s, c in sorted(sources.items()))
        print(f"    {name}: {breakdown}")

    # Write splits (without the 'source' key - clean chat format)
    for split_name, data in [
        ("sft_train", train_set),
        ("sft_val", val_set),
        ("sft_test", test_set),
    ]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                # Write only messages, not source metadata
                clean = {"messages": item["messages"]}
                f.write(json.dumps(clean) + "\n")
        print(f"  Wrote {path} ({len(data)} examples)")

    # Save test prompts separately for evaluation
    test_prompts_path = output_dir / "test_prompts.jsonl"
    with open(test_prompts_path, "w") as f:
        for item in test_set:
            f.write(json.dumps({"prompt": item["messages"][1]["content"]}) + "\n")
    print(f"  Wrote {test_prompts_path}")

    # Summary
    total = len(train_set) + len(val_set) + len(test_set)
    print(f"\nDone! {total} total examples across 3 splits.")
    return len(train_set), len(val_set), len(test_set)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip Manim render validation (use if data was pre-validated)"
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    prepare_dataset(
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_validation=args.skip_validation,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
