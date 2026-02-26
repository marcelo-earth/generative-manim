"""Validate existing datasets through the Manim verifier."""

import json
import sys
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
from rendering.manim_verifier import verify_code
from utils.code_extraction import clean_code

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
EXTRA_DATASET = Path(__file__).parent.parent / "dataset-2025-02-15T02-45-51-960Z.jsonl"


def load_all_examples():
    """Load all existing examples from repo."""
    examples = []

    # edoh dataset (use PEP8 formatted version with proper newlines)
    path = DATASETS_DIR / "edoh-dataset-format-pep8.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                msgs = data.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"), None)
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if user and assistant:
                    examples.append({"prompt": user, "code": assistant, "source": "edoh"})

    # physics dataset
    path = DATASETS_DIR / "physics-01.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                msgs = data.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"), None)
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if user and assistant:
                    examples.append({"prompt": user, "code": assistant, "source": "physics"})

    # Feb 2025 export
    if EXTRA_DATASET.exists():
        with open(EXTRA_DATASET) as f:
            for line in f:
                data = json.loads(line)
                msgs = data.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"), None)
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if user and assistant:
                    examples.append({"prompt": user, "code": assistant, "source": "feb2025"})

    # Code samples (no prompts, just code)
    code_dir = DATASETS_DIR / "code"
    if code_dir.exists():
        for py_file in sorted(code_dir.glob("*.py")):
            code = py_file.read_text()
            examples.append({
                "prompt": None,
                "code": code,
                "source": f"code/{py_file.name}",
            })

    return examples


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    examples = load_all_examples()
    print(f"Loaded {len(examples)} total examples")

    by_source = Counter(e["source"] for e in examples)
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")

    if limit:
        examples = examples[:limit]
        print(f"\nTesting first {limit} examples only")

    print(f"\nValidating {len(examples)} examples (1 at a time, be patient)...\n")

    stats = {"success": 0, "fail": 0}
    error_counts = Counter()
    source_stats = {}

    for i, ex in enumerate(examples):
        code = clean_code(ex["code"])
        result = verify_code(code, timeout=60)

        source = ex["source"]
        if source not in source_stats:
            source_stats[source] = {"success": 0, "fail": 0}

        if result.success:
            stats["success"] += 1
            source_stats[source]["success"] += 1
        else:
            stats["fail"] += 1
            source_stats[source]["fail"] += 1
            error_counts[result.error_type.value] += 1

        total = i + 1
        rate = stats["success"] / total * 100
        print(
            f"  [{total}/{len(examples)}] "
            f"{'OK' if result.success else 'FAIL':4s} | "
            f"{result.error_type.value:15s} | "
            f"{result.render_time:.1f}s | "
            f"{source:10s} | "
            f"{(ex['prompt'] or ex['source'])[:60]}"
        )

        # Print running stats every 25
        if total % 25 == 0:
            print(f"\n  --- Running: {stats['success']}/{total} ({rate:.0f}%) ---\n")

    # Final report
    total = stats["success"] + stats["fail"]
    rate = stats["success"] / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"RESULTS: {stats['success']}/{total} render successfully ({rate:.1f}%)")
    print(f"\nBy source:")
    for source, s in sorted(source_stats.items()):
        src_total = s["success"] + s["fail"]
        src_rate = s["success"] / src_total * 100 if src_total else 0
        print(f"  {source:15s}: {s['success']}/{src_total} ({src_rate:.0f}%)")
    print(f"\nError breakdown:")
    for error, count in error_counts.most_common():
        print(f"  {error}: {count}")


if __name__ == "__main__":
    main()
