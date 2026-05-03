<p align="center">
  <b>English</b> | <a href="README_CN.md">中文说明</a>
</p>

---

<p align="center">
  <img
    src=".github/logo.png"
    align="center"
    width="100"
    alt="Generative Manim"
    title="Generative Manim"
  />
  <h1 align="center">Generative Manim</h1>
</p>

<p align="center">
  🎨 GPT-4o powered generative videos. Concept. ⚡️ <a href="https://discord.gg/SNdbPU2AMM">Join our Discord server here!</a>
</p>

<p align="center">
  <a href="https://generative-manim.vercel.app">
    <img src="https://img.shields.io/static/v1?label=Demo&message=Generative%20Manim&color=000000&logo=vercel&style=flat" />
  </a>
  <a href="https://animo.video">
    <img src="https://img.shields.io/static/v1?label=Platform&message=Animo&color=E11D48&logo=openai&style=flat" />
  </a>
  <a href="">
    <img src="https://img.shields.io/static/v1?label=OpenAI%20API&message=GPT-4o&color=000000&logo=openai&style=flat" />
  </a>
  <a href="">
    <img src="https://img.shields.io/static/v1?label=Anthropic&message=Claude&color=000000&logo=anthropic&style=flat" />
  </a>
  <a href="./docs/featherless.md">
    <img src="https://img.shields.io/static/v1?label=Featherless&message=Open%20Models&color=6D28D9&style=flat" />
  </a>
</p>

---

![Preview](./.github/preview.jpg)

## 🚀 Concept

**Generative Manim** (GM) is a suite of tools that allows you to create videos with Manim using LLMs (Large Language Models) like GPT-4 or Claude. The idea is to enable anyone to create wonderful animations from text ✨.

It began as a prototype of a web app that uses [GPT-4](https://openai.com/research/gpt-4) to generate videos with [Manim](https://www.manim.community). The idea behind this project is taking advantage of the power of LLMs in programming, the understanding of human language and the animation capabilities of Manim to generate a tool that could be used by anyone to create videos. Regardless of their programming or video editing skills.

- 🖐️ [Generative Manim Demo](https://generative-manim.vercel.app/): Check out the demo of Generative Manim!
- 🔬 [Generative Manim API](https://github.com/360macky/generative-manim/tree/main/api): Build over the Animation Processing Interface, or API.
- ☁️ [Cloud Deployment Guide](./docs/cloud-deployment.md): Deploy the API on Render or another Docker-based cloud platform.
- 🧑‍💻 [Generative Manim Developers](https://discord.gg/SNdbPU2AMM): Join our Discord server, learn new things, share your creations and more!
- 🍎 [Generative Manim Streamlit (Legacy)](https://github.com/360macky/generative-manim/tree/main/streamlit): First LLM exploration of LLMs and Animation.

## 💻 Models

**Models** are the core of Generative Manim. A model is a way to convert text to code, that can later be rendered in a video.

| Name                          | Description                                                               | Engine                     | Phase |
| ----------------------------- | ------------------------------------------------------------------------- | -------------------------- | ----- |
| GM GPT-4o                     | Latest GPT model from OpenAI powered by a custom System Prompt            | GPT-4o                     | ✅    |
| GM GPT-3.5 Fine Tuned         | First Fine-tuned model of GPT-3.5                                         | GPT-3.5                    | ✅    |
| GM GPT-3.5 Physics Fine Tuned | Fine-tuned GPT-3.5 model trained to generate Physics animations           | GPT-3.5                    | ✅    |
| GM Claude Sonnet              | Claude Sonnet 3 model from Sonnet adapted with our custom System Prompt   | claude-3-sonnet-20240229   | ✅    |
| GM Claude Sonnet 3.5          | Claude Sonnet 3.5 model from Sonnet adapted with our custom System Prompt | claude-3-5-sonnet-20240620 | ✅    |
| GM Featherless Open Models    | OpenAI-compatible access to hosted open-weight models via Featherless     | Qwen, DeepSeek, CodeLlama, etc. | ✅ |
| GM Gemini 2.5 Flash           | Google's Gemini 2.5 Flash accessed via google-genai SDK                  | gemini-2.5-flash           | ✅    |
| GM Gemini 3 Flash             | Google's Gemini 3 Flash preview accessed via google-genai SDK            | gemini-3-flash-preview     | ✅    |
| GM Qwen 2.5 Coder 7B          | Open-source model fine-tuned with SFT + DPO + GRPO pipeline              | Qwen2.5-Coder-7B-Instruct | 🚧    |
| GM DeepSeek Coder V2 Lite      | Open-source model fine-tuned with SFT + DPO + GRPO pipeline              | DeepSeek-Coder-V2-Lite     | 🚧    |
| GM CodeLlama 7B                | Open-source model fine-tuned with SFT + DPO + GRPO pipeline              | CodeLlama-7b-Instruct      | 🚧    |

### 📡 New Models

If you want to suggest a new model, please open an issue in the [repository](https://github.com/360macky/generative-manim/issues) or talk with us in our [Discord server](https://discord.gg/SNdbPU2AMM).

## 🧠 Training Pipeline

We're training **open-source models** to generate Manim code using a 3-stage pipeline that distills from GPT-4o:

1. **SFT** (Supervised Fine-Tuning): Train on 5,000+ validated prompt→code pairs
2. **DPO** (Direct Preference Optimization): Learn from render success/failure pairs
3. **GRPO** (Group Relative Policy Optimization): RL with the Manim renderer as a deterministic reward signal

The key insight: Manim is a **deterministic verifier**: code either renders or crashes. This replaces the need for a reward model, similar to how DeepSeek-R1 uses math answer checkers.

**Base models**: Qwen 2.5 Coder 7B, DeepSeek Coder V2 Lite, CodeLlama 7B. All use QLoRA (4-bit) to fit on free Kaggle T4 GPUs.

## 📏 Benchmark

Generative Manim now includes an executable benchmark MVP for expert Manim code generation under [`training/benchmarks`](./training/benchmarks).

The benchmark is built around the right primitives for programming evaluation:

- a frozen task suite
- render-based scoring
- Manim-specific structural checks
- pass@k for stochastic code generation
- reproducible JSONL and JSON reports

Start here:

```bash
cd training
python -m benchmarks.run export \
  --suite benchmarks/tasks/core_v1.jsonl \
  --output ./outputs/benchmarks/core_v1_prompts.jsonl
```

Then use the generated prompt file with `python -m eval.generate_responses ...`, or run the full flow with:

```bash
bash ./scripts/run_benchmark.sh qwen2.5-coder-7b ./outputs/grpo/qwen2.5-coder-7b benchmarks/tasks/core_v1.jsonl grpo 5 0.8 1,5
```

See [`training/benchmarks/README.md`](./training/benchmarks/README.md) for the benchmark design and workflow.

Once you have multiple benchmark runs, compare them with:

```bash
cd training
python -m benchmarks.compare --results-dir ./outputs/benchmarks --suite core_v1
```

Or run a whole benchmark matrix from a manifest:

```bash
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/open_source_core_v1.json --dry-run
```

You can also benchmark hosted open-weight models through Featherless:

```bash
export FEATHERLESS_API_KEY="your-featherless-key"
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/featherless_core_v1.json --only qwen2.5-coder-7b-instruct-featherless
```

See [`docs/featherless.md`](./docs/featherless.md) for API usage, smoke tests, and the full Featherless benchmark workflow.

## ✨ Sponsors

**Generative Manim** is currently sponsored by **The Astronomical Software Company**.

## 🙌 Contributors

Thank you to everyone who has contributed to Generative Manim!

| Contributor | Contribution |
| ----------- | ------------ |
| [@abdullahsohaill](https://github.com/abdullahsohaill) | Add Gemini 2.5 Flash and Gemini 3 Flash support via google-genai |
| [@tranquac](https://github.com/tranquac) | Fix command injection vulnerability in ffmpeg video export |
| [@Wing900](https://github.com/Wing900) | Add Chinese README translation |

## 🤲 Contributing

Generative Manim is an open source project.

If you want to be the author of a new feature, fix a bug or contribute with something new.

Fork the repository and make changes as you like. [Pull requests](https://github.com/360macky/generative-manim/pulls) are warmly welcome. Remember you can also join our [Discord server](https://discord.gg/SNdbPU2AMM) to discuss new features, bugs or any other topic.
