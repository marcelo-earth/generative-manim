"""Generate benchmark responses with OpenAI-compatible hosted providers."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from utils.system_prompt import SYSTEM_PROMPT


DEFAULT_TEST = Path(__file__).parent.parent / "data" / "outputs" / "test_prompts.jsonl"
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"


def _provider_defaults(provider: str) -> tuple[str | None, str]:
    if provider == "featherless":
        return FEATHERLESS_BASE_URL, "FEATHERLESS_API_KEY"
    if provider == "openai":
        return None, "OPENAI_API_KEY"
    raise ValueError(f"Unsupported provider: {provider}")


def _load_prompts(test_path: Path, limit: int | None = None) -> list[dict]:
    prompts = []
    with open(test_path) as f:
        for line in f:
            if not line.strip():
                continue
            prompts.append(json.loads(line))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def generate_remote_responses(
    model: str,
    provider: str = "featherless",
    test_path: str | Path = DEFAULT_TEST,
    output_path: str | Path | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    samples_per_prompt: int = 1,
    seed: int | None = None,
    limit: int | None = None,
    max_retries: int = 3,
    retry_delay: float = 4.0,
) -> str:
    """Generate responses for benchmark prompts using a hosted chat model."""
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    if samples_per_prompt > 1 and temperature <= 0:
        raise ValueError("temperature must be > 0 when samples_per_prompt > 1")

    default_base_url, default_api_key_env = _provider_defaults(provider)
    base_url = base_url if base_url is not None else default_base_url
    api_key_env = api_key_env or default_api_key_env
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is required for provider={provider}")

    test_path = Path(test_path)
    prompts = _load_prompts(test_path, limit=limit)

    if output_path is None:
        safe_model = model.replace("/", "__")
        output_path = Path("outputs") / "remote" / provider / safe_model / "responses.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(
        f"Generating {samples_per_prompt} sample(s) for {len(prompts)} prompts "
        f"with {provider}:{model}"
    )

    with open(output_path, "w") as f:
        for row in tqdm(prompts, desc="Generating"):
            prompt = row["prompt"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            for sample_index in range(samples_per_prompt):
                request_payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if seed is not None:
                    request_payload["seed"] = seed + sample_index

                last_error = None
                started_at = time.perf_counter()
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(**request_payload)
                        latency = time.perf_counter() - started_at
                        content = response.choices[0].message.content or ""
                        payload = {
                            "prompt": prompt,
                            "response": content,
                            "sample_index": sample_index,
                            "provider": provider,
                            "model": model,
                            "latency_seconds": round(latency, 3),
                        }
                        for key in ("task_id", "category", "difficulty"):
                            if key in row:
                                payload[key] = row[key]
                        f.write(json.dumps(payload) + "\n")
                        f.flush()
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                        else:
                            raise RuntimeError(
                                f"Failed to generate sample {sample_index} for prompt: {prompt}"
                            ) from last_error

    print(f"Saved responses to {output_path}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark responses with an OpenAI-compatible API"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--provider", type=str, default="featherless", choices=["featherless", "openai"])
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=4.0)
    args = parser.parse_args()

    generate_remote_responses(
        model=args.model,
        provider=args.provider,
        test_path=args.test_path,
        output_path=args.output,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        samples_per_prompt=args.samples_per_prompt,
        seed=args.seed,
        limit=args.limit,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()
