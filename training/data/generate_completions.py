"""Generate Manim code completions from teacher model (GPT-4o) for each prompt."""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from ..utils.system_prompt import SYSTEM_PROMPT

DEFAULT_INPUT = Path(__file__).parent / "outputs" / "raw_prompts.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "raw_completions.jsonl"


async def generate_completion(
    client: AsyncOpenAI,
    prompt: str,
    category: str,
    model: str = "gpt-4o",
    semaphore: asyncio.Semaphore | None = None,
) -> dict | None:
    """Generate a single Manim code completion."""
    sem = semaphore or asyncio.Semaphore(5)
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            code = response.choices[0].message.content
            return {
                "category": category,
                "prompt": prompt,
                "completion": code,
                "model": model,
            }
        except Exception as e:
            print(f"Error generating completion: {e}")
            return None


async def generate_all_completions(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    model: str = "gpt-4o",
    concurrency: int = 5,
    resume: bool = True,
    limit: int | None = None,
):
    """Generate completions for all prompts."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = []
    with open(input_path) as f:
        for line in f:
            prompts.append(json.loads(line))

    if limit:
        prompts = prompts[:limit]

    # Resume support
    completed_prompts = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                completed_prompts.add(data["prompt"])
        print(f"Resuming: {len(completed_prompts)} completions already done")

    remaining = [p for p in prompts if p["prompt"] not in completed_prompts]
    print(f"Generating {len(remaining)} completions (model: {model})...")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    # Process in batches to write results incrementally
    batch_size = 50
    total_done = 0

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        tasks = [
            generate_completion(client, p["prompt"], p["category"], model, semaphore)
            for p in batch
        ]
        results = await asyncio.gather(*tasks)

        with open(output_path, "a") as f:
            for result in results:
                if result is not None:
                    f.write(json.dumps(result) + "\n")
                    total_done += 1

        print(f"  Progress: {total_done}/{len(remaining)}")

    print(f"Done. Total completions: {len(completed_prompts) + total_done}")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher completions")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(
        generate_all_completions(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            concurrency=args.concurrency,
            resume=not args.no_resume,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
