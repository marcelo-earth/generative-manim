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
</p>

---

![预览](./.github/preview.jpg)

## 🚀 概念

**Generative Manim** (GM) 是一套工具，允许你利用 GPT-4 或 Claude 等 LLM（大语言模型），通过 Manim 制作视频。其理念是让任何人都能通过文本创作出精彩的动画 ✨。

它最初是一个 Web 应用原型，利用 [GPT-4](https://openai.com/research/gpt-4) 结合 [Manim](https://www.manim.community) 生成视频。该项目背后的想法是利用 LLM 在编程方面的强大能力、对人类语言的理解力以及 Manim 的动画功能，生成一种任何人都可以使用的视频创作工具。无论其编程或视频剪辑技能如何。

- 🖐️ [Generative Manim 演示](https://generative-manim.vercel.app/): 查看 Generative Manim 的演示！
- 🔬 [Generative Manim API](https://github.com/360macky/generative-manim/tree/main/api): 基于动画处理接口（API）进行构建。
- 🧑‍💻 [Generative Manim 开发者](https://discord.gg/SNdbPU2AMM): 加入我们的 Discord 服务器，学习新知识，分享你的创作等等！
- 🍎 [Generative Manim Streamlit (旧版)](https://github.com/360macky/generative-manim/tree/main/streamlit): 对 LLM 和动画的首次探索。

## 💻 模型

**模型**是 Generative Manim 的核心。模型是一种将文本转换为代码的方式，随后这些代码可被渲染成视频。

| 名称 (Name)                   | 描述 (Description)                                           | 引擎 (Engine)              | 阶段 (Phase) |
| ----------------------------- | ------------------------------------------------------------ | -------------------------- | ------------ |
| GM GPT-4o                     | 由自定义系统提示词（System Prompt）驱动的 OpenAI 最新 GPT 模型 | GPT-4o                     | ✅            |
| GM GPT-3.5 Fine Tuned         | 首个 GPT-3.5 微调模型                                        | GPT-3.5                    | ✅            |
| GM GPT-3.5 Physics Fine Tuned | 经训练用于生成物理动画的 GPT-3.5 微调模型                    | GPT-3.5                    | ✅            |
| GM Claude Sonnet              | 经我们自定义系统提示词适配的 Claude Sonnet 3 模型            | claude-3-sonnet-20240229   | ✅            |
| GM Claude Sonnet 3.5          | 经我们自定义系统提示词适配的 Claude Sonnet 3.5 模型          | claude-3-5-sonnet-20240620 | ✅            |

### 📡 新模型

如果你想建议一个新模型，请在[仓库](https://github.com/360macky/generative-manim/issues)中提交 Issue，或者在我们的 [Discord 服务器](https://discord.gg/SNdbPU2AMM)中与我们交流。

## ✨ 赞助商

**Generative Manim** 目前由 **The Astronomical Software Company** 赞助。

## 🤲 贡献

Generative Manim 是一个开源项目。

如果你想开发新功能、修复 Bug 或贡献新内容。

请 Fork 本仓库并按需进行更改。热烈欢迎提交 [Pull requests](https://github.com/360macky/generative-manim/pulls)。记得你也可以加入我们的 [Discord 服务器](https://discord.gg/SNdbPU2AMM)来讨论新功能、Bug 或任何其他话题。
