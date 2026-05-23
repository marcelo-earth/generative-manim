<p align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文说明</a> | <b>Español</b> | <a href="README_DE.md">Deutsch</a> | <a href="README_KO.md">한국어</a>
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
  🎨 Videos generativos con GPT-4o. Concepto. ⚡️ <a href="https://discord.gg/SNdbPU2AMM">¡Únete a nuestro servidor de Discord aquí!</a>
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

![Vista previa](./.github/preview.jpg)

## 🚀 Concepto

**Generative Manim** (GM) es un conjunto de herramientas que te permite crear videos con Manim usando LLMs (Modelos de Lenguaje de Gran Escala) como GPT-4 o Claude. La idea es permitir que cualquier persona pueda crear animaciones maravillosas a partir de texto ✨.

Comenzó como un prototipo de una aplicación web que usa [GPT-4](https://openai.com/research/gpt-4) para generar videos con [Manim](https://www.manim.community). La idea detrás de este proyecto es aprovechar el poder de los LLMs en programación, la comprensión del lenguaje humano y las capacidades de animación de Manim para generar una herramienta que cualquier persona pueda usar para crear videos. Independientemente de sus habilidades de programación o edición de video.

- 🖐️ [Demo de Generative Manim](https://generative-manim.vercel.app/): ¡Prueba la demo de Generative Manim!
- 🔬 [API de Generative Manim](https://github.com/360macky/generative-manim/tree/main/api): Construye sobre la Interfaz de Procesamiento de Animaciones, o API.
- ☁️ [Guía de Despliegue en la Nube](./docs/cloud-deployment.md): Despliega la API en Render u otra plataforma cloud basada en Docker.
- 🧑‍💻 [Desarrolladores de Generative Manim](https://discord.gg/SNdbPU2AMM): ¡Únete a nuestro servidor de Discord, aprende cosas nuevas, comparte tus creaciones y más!
- 🍎 [Generative Manim Streamlit (Legado)](https://github.com/360macky/generative-manim/tree/main/streamlit): Primera exploración de LLMs y Animación.

## 💻 Modelos

Los **modelos** son el núcleo de Generative Manim. Un modelo es una forma de convertir texto en código, que luego puede renderizarse en un video.

| Nombre                        | Descripción                                                                       | Motor                      | Fase |
| ----------------------------- | --------------------------------------------------------------------------------- | -------------------------- | ----- |
| GM GPT-5.5                    | El modelo de frontera más reciente de OpenAI para trabajo profesional y de código  | gpt-5.5                    | ✅    |
| GM GPT-4o                     | Último modelo GPT de OpenAI impulsado por un System Prompt personalizado           | GPT-4o                     | ✅    |
| GM GPT-3.5 Fine Tuned         | Primer modelo ajustado fino de GPT-3.5                                            | GPT-3.5                    | ✅    |
| GM GPT-3.5 Physics Fine Tuned | Modelo GPT-3.5 ajustado fino para generar animaciones de Física                   | GPT-3.5                    | ✅    |
| GM Claude Sonnet              | Modelo Claude Sonnet 3 adaptado con nuestro System Prompt personalizado           | claude-3-sonnet-20240229   | ✅    |
| GM Claude Sonnet 3.5          | Modelo Claude Sonnet 3.5 adaptado con nuestro System Prompt personalizado         | claude-3-5-sonnet-20240620 | ✅    |
| GM Featherless Open Models    | Acceso compatible con OpenAI a modelos de código abierto alojados via Featherless | Qwen, DeepSeek, CodeLlama, etc. | ✅ |
| GM Gemini 2.5 Flash           | Gemini 2.5 Flash de Google accedido via google-genai SDK                         | gemini-2.5-flash           | ✅    |
| GM Gemini 3 Flash             | Vista previa de Gemini 3 Flash de Google accedido via google-genai SDK            | gemini-3-flash-preview     | ✅    |
| GM Qwen 2.5 Coder 7B          | Modelo de código abierto ajustado fino con pipeline SFT + DPO + GRPO             | Qwen2.5-Coder-7B-Instruct | 🚧    |
| GM DeepSeek Coder V2 Lite     | Modelo de código abierto ajustado fino con pipeline SFT + DPO + GRPO             | DeepSeek-Coder-V2-Lite     | 🚧    |
| GM CodeLlama 7B               | Modelo de código abierto ajustado fino con pipeline SFT + DPO + GRPO             | CodeLlama-7b-Instruct      | 🚧    |

### 📡 Nuevos Modelos

Si quieres sugerir un nuevo modelo, por favor abre un issue en el [repositorio](https://github.com/360macky/generative-manim/issues) o habla con nosotros en nuestro [servidor de Discord](https://discord.gg/SNdbPU2AMM).

## 🧠 Pipeline de Entrenamiento

Estamos entrenando **modelos de código abierto** para generar código Manim usando un pipeline de 3 etapas que destila de GPT-4o:

1. **SFT** (Ajuste Fino Supervisado): Entrena en más de 5,000 pares prompt→código validados
2. **DPO** (Optimización Directa de Preferencias): Aprende de pares de éxito/fallo de renderizado
3. **GRPO** (Optimización de Política Relativa de Grupo): RL con el renderizador de Manim como señal de recompensa determinista

La clave: Manim es un **verificador determinista**: el código o se renderiza o falla. Esto reemplaza la necesidad de un modelo de recompensa, similar a como DeepSeek-R1 usa verificadores de respuestas matemáticas.

**Modelos base**: Qwen 2.5 Coder 7B, DeepSeek Coder V2 Lite, CodeLlama 7B. Todos usan QLoRA (4-bit) para caber en las GPUs T4 de Kaggle gratuitas.

## 📏 Benchmark

Generative Manim ahora incluye un MVP de benchmark ejecutable para generación experta de código Manim en [`training/benchmarks`](./training/benchmarks).

El benchmark está construido sobre las primitivas correctas para evaluación de programación:

- un conjunto de tareas fijo
- puntuación basada en renderizado
- comprobaciones estructurales específicas de Manim
- pass@k para generación de código estocástica
- informes JSONL y JSON reproducibles

Comienza aquí:

```bash
cd training
python -m benchmarks.run export \
  --suite benchmarks/tasks/core_v1.jsonl \
  --output ./outputs/benchmarks/core_v1_prompts.jsonl
```

Luego usa el archivo de prompts generado con `python -m eval.generate_responses ...`, o ejecuta el flujo completo con:

```bash
bash ./scripts/run_benchmark.sh qwen2.5-coder-7b ./outputs/grpo/qwen2.5-coder-7b benchmarks/tasks/core_v1.jsonl grpo 5 0.8 1,5
```

Consulta [`training/benchmarks/README.md`](./training/benchmarks/README.md) para el diseño y flujo de trabajo del benchmark.

Una vez que tengas múltiples ejecuciones del benchmark, compáralas con:

```bash
cd training
python -m benchmarks.compare --results-dir ./outputs/benchmarks --suite core_v1
```

O ejecuta una matriz de benchmarks completa desde un manifiesto:

```bash
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/open_source_core_v1.json --dry-run
```

También puedes hacer benchmark de modelos de código abierto alojados a través de Featherless:

```bash
export FEATHERLESS_API_KEY="your-featherless-key"
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/featherless_core_v1.json --only qwen2.5-coder-7b-instruct-featherless
```

Consulta [`docs/featherless.md`](./docs/featherless.md) para el uso de la API, pruebas de humo y el flujo de trabajo completo del benchmark de Featherless.

## ✨ Patrocinadores

**Generative Manim** está actualmente patrocinado por **The Astronomical Software Company**.

## 🙌 Colaboradores

¡Gracias a todos los que han contribuido a Generative Manim!

| Colaborador | Contribución |
| ----------- | ------------ |
| [@abdullahsohaill](https://github.com/abdullahsohaill) | Añadir soporte para Gemini 2.5 Flash y Gemini 3 Flash via google-genai |
| [@tranquac](https://github.com/tranquac) | Corregir vulnerabilidad de inyección de comandos en la exportación de video ffmpeg |
| [@Wing900](https://github.com/Wing900) | Añadir traducción del README al chino |

## 🤲 Contribuir

Generative Manim es un proyecto de código abierto.

Si quieres ser el autor de una nueva funcionalidad, corregir un bug o contribuir con algo nuevo.

Haz un fork del repositorio y realiza los cambios que desees. Las [Pull requests](https://github.com/360macky/generative-manim/pulls) son bienvenidas. Recuerda que también puedes unirte a nuestro [servidor de Discord](https://discord.gg/SNdbPU2AMM) para discutir nuevas funcionalidades, bugs o cualquier otro tema.
