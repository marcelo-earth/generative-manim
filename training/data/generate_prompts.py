"""Use GPT-4o to expand seed prompts into ~8K diverse Manim prompts."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from .prompt_categories import CATEGORIES

EXPANSION_SYSTEM_PROMPT = """You are helping create training data for a Manim animation code generator.
Given a category and seed prompts, generate diverse NEW prompts that a user might ask.

Rules:
1. Each prompt should describe a Manim animation to create
2. Vary complexity: some simple (1-2 animations), some complex (5+ animations)
3. Be specific about colors, positions, timing, and effects
4. Include both technical (math/science) and creative prompts
5. Do NOT repeat the seed prompts
6. Output one prompt per line, no numbering or bullet points"""

DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "raw_prompts.jsonl"


async def expand_category(
    client: AsyncOpenAI,
    category: str,
    seeds: list[str],
    target_count: int = 530,
    model: str = "gpt-4o",
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict]:
    """Expand a single category from seeds to target_count prompts."""
    sem = semaphore or asyncio.Semaphore(5)
    results = []
    seeds_text = "\n".join(f"- {s}" for s in seeds)

    # Generate in batches of ~50
    batch_size = 50
    batches_needed = (target_count + batch_size - 1) // batch_size

    async def generate_batch(batch_idx: int) -> list[dict]:
        async with sem:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.9,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Category: {category}\n\n"
                            f"Seed prompts:\n{seeds_text}\n\n"
                            f"Generate {batch_size} new diverse Manim animation prompts "
                            f"for this category. Batch {batch_idx + 1}/{batches_needed}. "
                            f"Make each prompt unique and varied in complexity."
                        ),
                    },
                ],
            )

            lines = response.choices[0].message.content.strip().split("\n")
            return [
                {"category": category, "prompt": line.strip()}
                for line in lines
                if line.strip() and len(line.strip()) > 10
            ]

    tasks = [generate_batch(i) for i in range(batches_needed)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    for batch in batch_results:
        if isinstance(batch, list):
            results.extend(batch)

    return results[:target_count]


async def generate_all_prompts(
    output_path: str | Path = DEFAULT_OUTPUT,
    target_per_category: int = 530,
    model: str = "gpt-4o",
    concurrency: int = 5,
    resume: bool = True,
):
    """Generate prompts for all categories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing prompts
    existing = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                existing.add((data["category"], data["prompt"]))
        print(f"Resuming: {len(existing)} existing prompts found")

    # Check which categories need more prompts
    category_counts = {}
    for cat, prompt in existing:
        category_counts[cat] = category_counts.get(cat, 0) + 1

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    all_results = []

    for category, seeds in CATEGORIES.items():
        current = category_counts.get(category, 0)
        needed = target_per_category - current
        if needed <= 0:
            print(f"  {category}: already has {current} prompts, skipping")
            continue

        print(f"  {category}: generating {needed} more prompts...")
        results = await expand_category(
            client, category, seeds, needed, model, semaphore
        )

        # Filter duplicates
        new_results = []
        for r in results:
            key = (r["category"], r["prompt"])
            if key not in existing:
                existing.add(key)
                new_results.append(r)
        all_results.extend(new_results)

    # Append to output
    with open(output_path, "a") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

    total = len(existing)
    print(f"Total prompts: {total}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate Manim prompts")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-per-category", type=int, default=530)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        generate_all_prompts(
            output_path=args.output,
            target_per_category=args.target_per_category,
            model=args.model,
            concurrency=args.concurrency,
            resume=not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
