# Manim Prompt Context

The chat generation endpoint includes a compact Manim reference list in its system prompt. This gives the model a reminder of common Manim modules, classes, animations, mobjects, scenes, and utilities before it writes code.

## Where It Lives

The prompt context is stored in:

```text
api/prompts/manimDocs.py
```

`api/routes/chat_generation.py` imports it:

```python
from api.prompts.manimDocs import manimDocs
```

Then the endpoint inserts it into the system prompt under the `# Manim Library` section.

## What It Is

`manimDocs` is a Python string containing a Markdown-style reference index. It is not a live HTML fetch. The API does not download the Manim docs on every request.

The reference list helps the model by exposing the names of Manim APIs that are likely to be useful, such as:

- animation classes like `Create`, `FadeIn`, `Transform`, and `Rotate`
- mobjects like geometry, graphs, text, tables, and vector fields
- scene classes like `Scene`, `ThreeDScene`, and `MovingCameraScene`
- camera, configuration, and utility modules

This does not guarantee correct code, but it gives the model more local context than a plain prompt.

## Which Endpoint Uses It

The Manim docs context is used by:

```text
POST /v1/chat/generation
```

The simpler code generation endpoint:

```text
POST /v1/code/generation
```

uses its own shorter system prompt and does not currently include `manimDocs`.

## Updating The Context

The repository includes `docs.py`, which can crawl the public Manim documentation, convert pages to Markdown, and combine them into `combined_docs.md`.

The current `api/prompts/manimDocs.py` file is a curated compact index, not the full combined documentation. Keeping it compact matters because a full documentation dump can exceed model context limits and make requests slower or more expensive.

If you want to refresh the prompt context:

1. Use `docs.py` to regenerate local Markdown from the Manim docs.
2. Extract the compact reference sections that are useful for code generation.
3. Replace or edit the string in `api/prompts/manimDocs.py`.
4. Test `/v1/chat/generation` with several prompts and verify that the generated code still renders.

## Why Not Include Every HTML Page?

Including all HTML or Markdown documentation in every request has tradeoffs:

- it can exceed token limits
- it increases latency and cost
- it can distract the model with irrelevant details
- it makes prompt behavior harder to reason about

For production, a better long-term approach is retrieval: index the docs, search only the relevant sections for the user's prompt, and inject those sections into the request.
