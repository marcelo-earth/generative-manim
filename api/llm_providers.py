import os
from typing import Iterator

try:
    from google import genai
except ImportError:
    genai = None


GEMINI_MODEL_ALIASES = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
}


def normalize_gemini_model(model: str) -> str:
    return GEMINI_MODEL_ALIASES.get(model, model)


def strip_markdown_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def generate_gemini_content(model: str, system_prompt: str, prompt: str) -> str:
    if genai is None:
        raise RuntimeError("google-genai is required to use Gemini models")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to use Gemini models")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=normalize_gemini_model(model),
        contents=f"{system_prompt.strip()}\n\nUser request: {prompt}",
    )
    return strip_markdown_code_fence(response.text)


def generate_gemini_content_stream(
    model: str, system_prompt: str, messages: list
) -> Iterator[str]:
    """Yield text chunks from a Gemini streaming response.

    *messages* is a list of ``{"role": ..., "content": ...}`` dicts. The
    system prompt is prepended as a plain string so Gemini receives the full
    conversation context (the API does not support a separate system role).
    """
    if genai is None:
        raise RuntimeError("google-genai is required to use Gemini models")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to use Gemini models")

    client = genai.Client(api_key=api_key)

    parts = [system_prompt.strip()]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        parts.append(f"{role.capitalize()}: {content}")

    contents = "\n\n".join(parts)

    for chunk in client.models.generate_content_stream(
        model=normalize_gemini_model(model),
        contents=contents,
    ):
        if chunk.text:
            yield chunk.text
