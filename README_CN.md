<p align="center">
  <a href="README.md">English</a> | <b>中文说明</b>
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
  🎨 由 GPT-4o 驱动的生成式视频。概念验证。⚡️ <a href="https://discord.gg/SNdbPU2AMM">点击此处加入我们的 Discord 服务器！</a>
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

![预览](./.github/preview.jpg)

## 🚀 概念

**Generative Manim** (GM) 是一套工具，允许你利用 GPT-4 或 Claude 等 LLM（大语言模型），通过 Manim 制作视频。其理念是让任何人都能通过文本创作出精彩的动画 ✨。

它最初是一个 Web 应用原型，利用 [GPT-4](https://openai.com/research/gpt-4) 结合 [Manim](https://www.manim.community) 生成视频。该项目背后的想法是利用 LLM 在编程方面的强大能力、对人类语言的理解力以及 Manim 的动画功能，生成一种任何人都可以使用的视频创作工具。无论其编程或视频剪辑技能如何。

- 🖐️ [Generative Manim 演示](https://generative-manim.vercel.app/): 查看 Generative Manim 的演示！
- 🔬 [Generative Manim API](https://github.com/360macky/generative-manim/tree/main/api): 基于动画处理接口（API）进行构建。
- ☁️ [云端部署指南](./docs/cloud-deployment.md): 在 Render 或其他基于 Docker 的云平台上部署 API。
- 📚 [文档索引](./docs/README.md): 浏览所有详细指南。
- 🧑‍💻 [Generative Manim 开发者](https://discord.gg/SNdbPU2AMM): 加入我们的 Discord 服务器，学习新知识，分享你的创作等等！
- 🍎 [Generative Manim Streamlit (旧版)](https://github.com/360macky/generative-manim/tree/main/streamlit): 对 LLM 和动画的首次探索。

## 💻 模型

**模型**是 Generative Manim 的核心。模型是一种将文本转换为代码的方式，随后这些代码可被渲染成视频。

| 名称 (Name)                   | 描述 (Description)                                           | 引擎 (Engine)              | 阶段 (Phase) |
| ----------------------------- | ------------------------------------------------------------ | -------------------------- | ------------ |
| GM GPT-5.5                    | OpenAI 用于复杂专业及编码工作的最新前沿模型                  | gpt-5.5                    | ✅            |
| GM GPT-4o                     | 由自定义系统提示词（System Prompt）驱动的 OpenAI 最新 GPT 模型 | GPT-4o                     | ✅            |
| GM GPT-3.5 Fine Tuned         | 首个 GPT-3.5 微调模型                                        | GPT-3.5                    | ✅            |
| GM GPT-3.5 Physics Fine Tuned | 经训练用于生成物理动画的 GPT-3.5 微调模型                    | GPT-3.5                    | ✅            |
| GM Claude Sonnet              | 经我们自定义系统提示词适配的 Claude Sonnet 3 模型            | claude-3-sonnet-20240229   | ✅            |
| GM Claude Sonnet 3.5          | 经我们自定义系统提示词适配的 Claude Sonnet 3.5 模型          | claude-3-5-sonnet-20240620 | ✅            |
| GM Featherless Open Models    | 通过 Featherless 以 OpenAI 兼容的方式访问的托管开源权重模型 | Qwen, DeepSeek, CodeLlama 等 | ✅          |
| GM Gemini 2.5 Flash           | 通过 google-genai SDK 访问的 Google Gemini 2.5 Flash 模型   | gemini-2.5-flash           | ✅            |
| GM Gemini 3 Flash             | 通过 google-genai SDK 访问的 Google Gemini 3 Flash 预览版   | gemini-3-flash-preview     | ✅            |
| GM Qwen 2.5 Coder 7B          | 使用 SFT + DPO + GRPO 流程进行微调的开源模型                | Qwen2.5-Coder-7B-Instruct  | 🚧            |
| GM DeepSeek Coder V2 Lite     | 使用 SFT + DPO + GRPO 流程进行微调的开源模型                | DeepSeek-Coder-V2-Lite     | 🚧            |
| GM CodeLlama 7B               | 使用 SFT + DPO + GRPO 流程进行微调的开源模型                | CodeLlama-7b-Instruct      | 🚧            |

### 📡 新模型

如果你想建议一个新模型，请在[仓库](https://github.com/360macky/generative-manim/issues)中提交 Issue，或者在我们的 [Discord 服务器](https://discord.gg/SNdbPU2AMM)中与我们交流。

## 🧠 训练流程

我们正在训练**开源模型**来生成 Manim 代码，采用一个从 GPT-4o 蒸馏的三阶段流程:

1. **SFT**（监督式微调）: 在 5,000 多对经过验证的 prompt 与 code 上进行训练。
2. **DPO**（直接偏好优化）: 从渲染成功与失败的样本对中学习偏好。
3. **GRPO**（群组相对策略优化）: 使用 Manim 渲染器作为确定性奖励信号的强化学习方法。

关键洞察是: Manim 是一个**确定性的验证器**, 代码要么能成功渲染, 要么会崩溃。这取代了对奖励模型的需求, 思路与 DeepSeek-R1 使用数学答案校验器的做法类似。

**基础模型**: Qwen 2.5 Coder 7B, DeepSeek Coder V2 Lite, CodeLlama 7B。全部使用 QLoRA（4-bit）以适配免费的 Kaggle T4 GPU。

## 📏 基准测试

Generative Manim 现在在 [`training/benchmarks`](./training/benchmarks) 中提供了一个可执行的基准测试 MVP，用于专业级的 Manim 代码生成评估。

该基准测试围绕编程评估应有的核心要素构建:

- 一个固定的任务集
- 基于渲染的评分
- Manim 特定的结构性检查
- 针对随机性代码生成的 pass@k 指标
- 可复现的 JSONL 与 JSON 报告

入门:

```bash
cd training
python -m benchmarks.run export \
  --suite benchmarks/tasks/core_v1.jsonl \
  --output ./outputs/benchmarks/core_v1_prompts.jsonl
```

然后使用生成的 prompt 文件运行 `python -m eval.generate_responses ...`, 或者直接运行完整流程:

```bash
bash ./scripts/run_benchmark.sh qwen2.5-coder-7b ./outputs/grpo/qwen2.5-coder-7b benchmarks/tasks/core_v1.jsonl grpo 5 0.8 1,5
```

更多基准测试的设计与工作流程, 请参阅 [`training/benchmarks/README.md`](./training/benchmarks/README.md)。

得到多次基准运行结果后, 可以使用以下命令进行对比:

```bash
cd training
python -m benchmarks.compare --results-dir ./outputs/benchmarks --suite core_v1
```

也可以根据 manifest 运行完整的基准矩阵:

```bash
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/open_source_core_v1.json --dry-run
```

你还可以通过 Featherless 对托管的开源权重模型进行基准测试:

```bash
export FEATHERLESS_API_KEY="your-featherless-key"
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/featherless_core_v1.json --only qwen2.5-coder-7b-instruct-featherless
```

API 用法, 冒烟测试以及完整的 Featherless 基准工作流程, 请参阅 [`docs/featherless.md`](./docs/featherless.md)。

## ✨ 赞助商

**Generative Manim** 目前由 **The Astronomical Software Company** 赞助。

## 🙌 贡献者

感谢每一位为 Generative Manim 做出贡献的人！

| 贡献者 | 贡献内容 |
| ----------- | ------------ |
| [@abdullahsohaill](https://github.com/abdullahsohaill) | 通过 google-genai 增加 Gemini 2.5 Flash 与 Gemini 3 Flash 支持 |
| [@tranquac](https://github.com/tranquac) | 修复 ffmpeg 视频导出中的命令注入漏洞 |
| [@Wing900](https://github.com/Wing900) | 增加中文 README 翻译 |

## 🤲 贡献

Generative Manim 是一个开源项目。

如果你想开发新功能、修复 Bug 或贡献新内容。

请 Fork 本仓库并按需进行更改。热烈欢迎提交 [Pull requests](https://github.com/360macky/generative-manim/pulls)。记得你也可以加入我们的 [Discord 服务器](https://discord.gg/SNdbPU2AMM)来讨论新功能、Bug 或任何其他话题。
