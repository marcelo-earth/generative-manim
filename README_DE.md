<p align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文说明</a> | <a href="README_ES.md">Español</a> | <b>Deutsch</b> | <a href="README_KO.md">한국어</a>
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
  🎨 GPT-4o-gestützte generative Videos. Konzept. ⚡️ <a href="https://discord.gg/SNdbPU2AMM">Tritt unserem Discord-Server bei!</a>
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

![Vorschau](./.github/preview.jpg)

## 🚀 Konzept

**Generative Manim** (GM) ist eine Suite von Werkzeugen, die es ermöglicht, Videos mit Manim mithilfe von LLMs (Große Sprachmodelle) wie GPT-4 oder Claude zu erstellen. Die Idee ist, jedem zu ermöglichen, wunderbare Animationen aus Text zu erstellen ✨.

Es begann als Prototyp einer Web-App, die [GPT-4](https://openai.com/research/gpt-4) verwendet, um Videos mit [Manim](https://www.manim.community) zu generieren. Die Idee hinter diesem Projekt ist es, die Stärke von LLMs in der Programmierung, das Verständnis menschlicher Sprache und die Animationsfähigkeiten von Manim zu nutzen, um ein Werkzeug zu schaffen, das jeder verwenden kann, um Videos zu erstellen. Unabhängig von seinen Programmier- oder Videobearbeitungskenntnissen.

- 🖐️ [Generative Manim Demo](https://generative-manim.vercel.app/): Schau dir die Demo von Generative Manim an!
- 🔬 [Generative Manim API](https://github.com/360macky/generative-manim/tree/main/api): Baue auf der Animation Processing Interface, oder API, auf.
- ☁️ [Cloud-Deployment-Leitfaden](./docs/cloud-deployment.md): Deploye die API auf Render oder einer anderen Docker-basierten Cloud-Plattform.
- 🧑‍💻 [Generative Manim Entwickler](https://discord.gg/SNdbPU2AMM): Tritt unserem Discord-Server bei, lerne Neues, teile deine Kreationen und mehr!
- 🍎 [Generative Manim Streamlit (Legacy)](https://github.com/360macky/generative-manim/tree/main/streamlit): Erste LLM-Erkundung von LLMs und Animation.

## 💻 Modelle

**Modelle** sind der Kern von Generative Manim. Ein Modell ist eine Möglichkeit, Text in Code zu konvertieren, der später in ein Video gerendert werden kann.

| Name                          | Beschreibung                                                                     | Engine                     | Phase |
| ----------------------------- | -------------------------------------------------------------------------------- | -------------------------- | ----- |
| GM GPT-5.5                    | OpenAIs neuestes Frontier-Modell für komplexe professionelle und Coding-Arbeit   | gpt-5.5                    | ✅    |
| GM GPT-4o                     | Neuestes GPT-Modell von OpenAI mit einem benutzerdefinierten System Prompt       | GPT-4o                     | ✅    |
| GM GPT-3.5 Fine Tuned         | Erstes Fine-tuned-Modell von GPT-3.5                                            | GPT-3.5                    | ✅    |
| GM GPT-3.5 Physics Fine Tuned | Fine-tuned GPT-3.5-Modell, trainiert zur Generierung von Physik-Animationen      | GPT-3.5                    | ✅    |
| GM Claude Sonnet              | Claude Sonnet 3-Modell, angepasst mit unserem benutzerdefinierten System Prompt  | claude-3-sonnet-20240229   | ✅    |
| GM Claude Sonnet 3.5          | Claude Sonnet 3.5-Modell, angepasst mit unserem benutzerdefinierten System Prompt | claude-3-5-sonnet-20240620 | ✅    |
| GM Featherless Open Models    | OpenAI-kompatibler Zugang zu gehosteten Open-Weight-Modellen via Featherless     | Qwen, DeepSeek, CodeLlama, etc. | ✅ |
| GM Gemini 2.5 Flash           | Googles Gemini 2.5 Flash, aufgerufen via google-genai SDK                       | gemini-2.5-flash           | ✅    |
| GM Gemini 3 Flash             | Googles Gemini 3 Flash Preview, aufgerufen via google-genai SDK                 | gemini-3-flash-preview     | ✅    |
| GM Qwen 2.5 Coder 7B          | Open-Source-Modell, feinabgestimmt mit SFT + DPO + GRPO-Pipeline               | Qwen2.5-Coder-7B-Instruct | 🚧    |
| GM DeepSeek Coder V2 Lite     | Open-Source-Modell, feinabgestimmt mit SFT + DPO + GRPO-Pipeline               | DeepSeek-Coder-V2-Lite     | 🚧    |
| GM CodeLlama 7B               | Open-Source-Modell, feinabgestimmt mit SFT + DPO + GRPO-Pipeline               | CodeLlama-7b-Instruct      | 🚧    |

### 📡 Neue Modelle

Wenn du ein neues Modell vorschlagen möchtest, öffne bitte ein Issue im [Repository](https://github.com/360macky/generative-manim/issues) oder sprich mit uns auf unserem [Discord-Server](https://discord.gg/SNdbPU2AMM).

## 🧠 Trainings-Pipeline

Wir trainieren **Open-Source-Modelle** zur Generierung von Manim-Code mithilfe einer 3-stufigen Pipeline, die aus GPT-4o destilliert:

1. **SFT** (Supervised Fine-Tuning): Training auf 5.000+ validierten Prompt-Code-Paaren
2. **DPO** (Direct Preference Optimization): Lernen aus Render-Erfolgs-/Fehlschlag-Paaren
3. **GRPO** (Group Relative Policy Optimization): RL mit dem Manim-Renderer als deterministisches Belohnungssignal

Die wichtigste Erkenntnis: Manim ist ein **deterministischer Verifier**: Code rendert entweder oder stürzt ab. Dies ersetzt die Notwendigkeit eines Belohnungsmodells, ähnlich wie DeepSeek-R1 mathematische Antwortprüfer verwendet.

**Basismodelle**: Qwen 2.5 Coder 7B, DeepSeek Coder V2 Lite, CodeLlama 7B. Alle verwenden QLoRA (4-bit), um auf kostenlose Kaggle T4 GPUs zu passen.

## 📏 Benchmark

Generative Manim enthält jetzt ein ausführbares Benchmark-MVP für die Expertengenerierung von Manim-Code unter [`training/benchmarks`](./training/benchmarks).

Das Benchmark ist um die richtigen Primitive für die Programmierevaluierung aufgebaut:

- eine gefrorene Aufgabensammlung
- renderbasierte Bewertung
- Manim-spezifische Strukturprüfungen
- pass@k für stochastische Code-Generierung
- reproduzierbare JSONL- und JSON-Berichte

Hier starten:

```bash
cd training
python -m benchmarks.run export \
  --suite benchmarks/tasks/core_v1.jsonl \
  --output ./outputs/benchmarks/core_v1_prompts.jsonl
```

Dann verwende die generierte Prompt-Datei mit `python -m eval.generate_responses ...`, oder führe den vollständigen Ablauf mit folgendem Befehl aus:

```bash
bash ./scripts/run_benchmark.sh qwen2.5-coder-7b ./outputs/grpo/qwen2.5-coder-7b benchmarks/tasks/core_v1.jsonl grpo 5 0.8 1,5
```

Sieh [`training/benchmarks/README.md`](./training/benchmarks/README.md) für das Benchmark-Design und den Workflow.

Sobald du mehrere Benchmark-Läufe hast, vergleiche sie mit:

```bash
cd training
python -m benchmarks.compare --results-dir ./outputs/benchmarks --suite core_v1
```

Oder führe eine vollständige Benchmark-Matrix aus einem Manifest aus:

```bash
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/open_source_core_v1.json --dry-run
```

Du kannst auch gehostete Open-Weight-Modelle über Featherless benchmarken:

```bash
export FEATHERLESS_API_KEY="your-featherless-key"
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/featherless_core_v1.json --only qwen2.5-coder-7b-instruct-featherless
```

Sieh [`docs/featherless.md`](./docs/featherless.md) für API-Nutzung, Smoke-Tests und den vollständigen Featherless-Benchmark-Workflow.

## ✨ Sponsoren

**Generative Manim** wird derzeit von **The Astronomical Software Company** gesponsert.

## 🙌 Mitwirkende

Vielen Dank an alle, die zu Generative Manim beigetragen haben!

| Mitwirkende(r) | Beitrag |
| -------------- | ------- |
| [@abdullahsohaill](https://github.com/abdullahsohaill) | Gemini 2.5 Flash und Gemini 3 Flash Unterstützung via google-genai hinzugefügt |
| [@tranquac](https://github.com/tranquac) | Command-Injection-Schwachstelle beim ffmpeg-Videoexport behoben |
| [@Wing900](https://github.com/Wing900) | Chinesische README-Übersetzung hinzugefügt |

## 🤲 Beitragen

Generative Manim ist ein Open-Source-Projekt.

Wenn du Autor einer neuen Funktion sein, einen Bug beheben oder etwas Neues beitragen möchtest.

Forke das Repository und nimm Änderungen nach Belieben vor. [Pull Requests](https://github.com/360macky/generative-manim/pulls) sind herzlich willkommen. Denke daran, dass du auch unserem [Discord-Server](https://discord.gg/SNdbPU2AMM) beitreten kannst, um neue Funktionen, Bugs oder andere Themen zu besprechen.
