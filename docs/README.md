# Generative Manim Docs

This directory contains the long-form guides that complement the project [README](../README.md). Each guide focuses on a single topic so they can be read independently.

## Guides

| Guide | What it covers |
| ----- | -------------- |
| [Cloud Deployment](./cloud-deployment.md) | Production deployment of the API on Render, Fly.io, Railway, Google Cloud Run, AWS ECS, or any Docker-based platform. Includes recommended architecture, environment variables, and storage notes. |
| [Featherless](./featherless.md) | Using [Featherless](https://featherless.ai) as an OpenAI-compatible provider for open-weight models, both for API code generation and for benchmarking with the render-based Manim verifier. |
| [Manim Prompt Context](./manim-prompt-context.md) | How the chat generation endpoint injects a compact Manim reference list (`api/prompts/manimDocs.py`) into its system prompt. |

## Related Documentation

- [Root README](../README.md) — project overview, model list, and benchmark entry points.
- [API README](../api/README.md) — local setup, endpoints, and usage examples for the Animation Processing Interface.
- [Training README](../training/README.md) — SFT, DPO, and GRPO training pipeline for open-source Manim models.
- [Benchmarks README](../training/benchmarks/README.md) — benchmark design and workflow for evaluating models on Manim code generation.
