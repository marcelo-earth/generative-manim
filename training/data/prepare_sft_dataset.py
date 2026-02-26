"""Merge synthetic + existing data, normalize format, split into train/val/test."""

import argparse
import json
import random
from pathlib import Path

from ..utils.code_extraction import normalize_class_name, ensure_manim_import
from ..utils.system_prompt import SYSTEM_PROMPT

DEFAULT_VALIDATED = Path(__file__).parent / "outputs" / "validated_completions.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"
DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"


def load_existing_datasets() -> list[dict]:
    """Load existing edoh and physics datasets."""
    examples = []

    # Load edoh dataset
    edoh_path = DATASETS_DIR / "edoh-dataset.jsonl"
    if edoh_path.exists():
        with open(edoh_path) as f:
            for line in f:
                data = json.loads(line)
                messages = data.get("messages", [])
                # Extract user prompt and assistant code
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"), None
                )
                assistant_msg = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), None
                )
                if user_msg and assistant_msg:
                    examples.append(
                        {
                            "prompt": user_msg,
                            "code": assistant_msg,
                            "source": "edoh",
                        }
                    )
        print(f"  Loaded {len(examples)} from edoh-dataset.jsonl")

    # Load physics dataset
    physics_count = 0
    physics_path = DATASETS_DIR / "physics-01.jsonl"
    if physics_path.exists():
        with open(physics_path) as f:
            for line in f:
                data = json.loads(line)
                messages = data.get("messages", [])
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"), None
                )
                assistant_msg = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), None
                )
                if user_msg and assistant_msg:
                    examples.append(
                        {
                            "prompt": user_msg,
                            "code": assistant_msg,
                            "source": "physics",
                        }
                    )
                    physics_count += 1
        print(f"  Loaded {physics_count} from physics-01.jsonl")

    return examples


def normalize_example(example: dict) -> dict:
    """Normalize a single example: fix class name, imports, format."""
    code = example["code"]

    # Normalize class name to GenScene
    code = normalize_class_name(code)
    code = ensure_manim_import(code)

    # Format as chat messages
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": code},
        ]
    }


def prepare_dataset(
    validated_path: str | Path = DEFAULT_VALIDATED,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    test_size: int = 200,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Merge, normalize, and split dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []

    # Load validated synthetic data
    validated_path = Path(validated_path)
    if validated_path.exists():
        with open(validated_path) as f:
            for line in f:
                data = json.loads(line)
                all_examples.append(
                    {
                        "prompt": data["prompt"],
                        "code": data["code"],
                        "source": "synthetic",
                    }
                )
        print(f"  Loaded {len(all_examples)} validated synthetic examples")

    # Load existing datasets
    existing = load_existing_datasets()
    all_examples.extend(existing)

    print(f"\n  Total examples: {len(all_examples)}")

    # Normalize all examples
    normalized = [normalize_example(ex) for ex in all_examples]

    # Deduplicate by prompt
    seen = set()
    deduped = []
    for ex in normalized:
        prompt = ex["messages"][1]["content"]
        if prompt not in seen:
            seen.add(prompt)
            deduped.append(ex)
    print(f"  After dedup: {len(deduped)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(deduped)

    test_set = deduped[:test_size]
    remaining = deduped[test_size:]
    val_size = int(len(remaining) * val_ratio)
    val_set = remaining[:val_size]
    train_set = remaining[val_size:]

    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # Write splits
    for name, data in [
        ("sft_train", train_set),
        ("sft_val", val_set),
        ("sft_test", test_set),
    ]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Wrote {path}")

    # Also save test prompts separately for evaluation
    test_prompts_path = output_dir / "test_prompts.jsonl"
    with open(test_prompts_path, "w") as f:
        for item in test_set:
            f.write(
                json.dumps({"prompt": item["messages"][1]["content"]}) + "\n"
            )

    return len(train_set), len(val_set), len(test_set)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset")
    parser.add_argument("--validated", type=str, default=str(DEFAULT_VALIDATED))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        validated_path=args.validated,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
