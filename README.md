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
</p>

---

![Preview](./.github/preview.jpg)

## 🚀 Concept

**Generative Manim** (GM) is a suite of tools that allows you to create videos with Manim using LLMs (Large Language Models) like GPT-4 or Claude. The idea is to enable anyone to create wonderful animations from text ✨.

It began as a prototype of a web app that uses [GPT-4](https://openai.com/research/gpt-4) to generate videos with [Manim](https://www.manim.community). The idea behind this project is taking advantage of the power of LLMs in programming, the understanding of human language and the animation capabilities of Manim to generate a tool that could be used by anyone to create videos. Regardless of their programming or video editing skills.

- 🖐️ [Generative Manim Demo](https://generative-manim.vercel.app/): Check out the demo of Generative Manim!
- 🔬 [Generative Manim API](https://github.com/360macky/generative-manim/tree/main/api): Build over the Animation Processing Interface, or API.
- 🧑‍💻 [Generative Manim Developers](https://discord.gg/HkbYEGybGv): Join our Discord server, learn new things, share your creations and more!
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
| GM Qwen 2.5 Coder 7B          | Open-source model fine-tuned with SFT + DPO + GRPO pipeline              | Qwen2.5-Coder-7B-Instruct | 🚧    |
| GM DeepSeek Coder V2 Lite      | Open-source model fine-tuned with SFT + DPO + GRPO pipeline              | DeepSeek-Coder-V2-Lite     | 🚧    |
| GM CodeLlama 7B                | Open-source model fine-tuned with SFT + DPO + GRPO pipeline              | CodeLlama-7b-Instruct      | 🚧    |

### 📡 New Models

If you want to suggest a new model, please open an issue in the [repository](https://github.com/360macky/generative-manim/issues) or talk with us in our [Discord server](https://discord.gg/HkbYEGybGv).

## 🧠 Training Pipeline

We're training **open-source models** to generate Manim code using a 3-stage pipeline that distills from GPT-4o:

1. **SFT** (Supervised Fine-Tuning) — Train on 5,000+ validated prompt→code pairs
2. **DPO** (Direct Preference Optimization) — Learn from render success/failure pairs
3. **GRPO** (Group Relative Policy Optimization) — RL with the Manim renderer as a deterministic reward signal

The key insight: Manim is a **deterministic verifier** — code either renders or crashes. This replaces the need for a reward model, similar to how DeepSeek-R1 uses math answer checkers.

**Base models**: Qwen 2.5 Coder 7B, DeepSeek Coder V2 Lite, CodeLlama 7B — all using QLoRA (4-bit) to fit on free Kaggle T4 GPUs.

## ✨ Sponsors

**Generative Manim** is currently sponsored by **The Astronomical Software Company**.

## 🤲 Contributing

Generative Manim is an open source project.

If you want to be the author of a new feature, fix a bug or contribute with something new.

Fork the repository and make changes as you like. [Pull requests](https://github.com/360macky/generative-manim/pulls) are warmly welcome. Remember you can also join our [Discord server](https://discord.gg/HkbYEGybGv) to discuss new features, bugs or any other topic.
