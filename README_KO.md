<p align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文说明</a> | <a href="README_ES.md">Español</a> | <a href="README_DE.md">Deutsch</a> | <b>한국어</b>
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
  🎨 GPT-4o 기반 생성형 비디오. 개념 증명. ⚡️ <a href="https://discord.gg/SNdbPU2AMM">Discord 서버에 참여하세요!</a>
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

![미리보기](./.github/preview.jpg)

## 🚀 개념

**Generative Manim** (GM)은 GPT-4나 Claude와 같은 LLM(대형 언어 모델)을 사용하여 Manim으로 비디오를 만들 수 있는 도구 모음입니다. 누구나 텍스트에서 멋진 애니메이션을 만들 수 있도록 하는 것이 목표입니다 ✨.

[GPT-4](https://openai.com/research/gpt-4)를 사용하여 [Manim](https://www.manim.community)으로 비디오를 생성하는 웹 앱 프로토타입으로 시작되었습니다. 이 프로젝트의 아이디어는 프로그래밍에서 LLM의 힘, 인간 언어에 대한 이해, 그리고 Manim의 애니메이션 기능을 활용하여 누구나 비디오를 만들 수 있는 도구를 생성하는 것입니다. 프로그래밍이나 영상 편집 기술에 상관없이.

- 🖐️ [Generative Manim 데모](https://generative-manim.vercel.app/): Generative Manim 데모를 확인해보세요!
- 🔬 [Generative Manim API](https://github.com/360macky/generative-manim/tree/main/api): 애니메이션 처리 인터페이스(API)를 기반으로 구축하세요.
- ☁️ [클라우드 배포 가이드](./docs/cloud-deployment.md): Render 또는 다른 Docker 기반 클라우드 플랫폼에 API를 배포하세요.
- 🧑‍💻 [Generative Manim 개발자](https://discord.gg/SNdbPU2AMM): Discord 서버에 참여하여 새로운 것을 배우고, 창작물을 공유하세요!
- 🍎 [Generative Manim Streamlit (레거시)](https://github.com/360macky/generative-manim/tree/main/streamlit): LLM과 애니메이션의 첫 번째 탐구.

## 💻 모델

**모델**은 Generative Manim의 핵심입니다. 모델은 텍스트를 코드로 변환하는 방법으로, 나중에 비디오로 렌더링될 수 있습니다.

| 이름                          | 설명                                                                          | 엔진                       | 단계 |
| ----------------------------- | ----------------------------------------------------------------------------- | -------------------------- | ----- |
| GM GPT-5.5                    | 복잡한 전문적 및 코딩 작업을 위한 OpenAI의 최신 프론티어 모델                  | gpt-5.5                    | ✅    |
| GM GPT-4o                     | 커스텀 System Prompt로 구동되는 OpenAI의 최신 GPT 모델                         | GPT-4o                     | ✅    |
| GM GPT-3.5 Fine Tuned         | 첫 번째 GPT-3.5 파인튜닝 모델                                                 | GPT-3.5                    | ✅    |
| GM GPT-3.5 Physics Fine Tuned | 물리 애니메이션 생성을 위해 훈련된 파인튜닝 GPT-3.5 모델                       | GPT-3.5                    | ✅    |
| GM Claude Sonnet              | 커스텀 System Prompt로 적용된 Claude Sonnet 3 모델                            | claude-3-sonnet-20240229   | ✅    |
| GM Claude Sonnet 3.5          | 커스텀 System Prompt로 적용된 Claude Sonnet 3.5 모델                          | claude-3-5-sonnet-20240620 | ✅    |
| GM Featherless Open Models    | Featherless를 통해 호스팅된 오픈 웨이트 모델에 대한 OpenAI 호환 접근          | Qwen, DeepSeek, CodeLlama, 등 | ✅ |
| GM Gemini 2.5 Flash           | google-genai SDK를 통해 접근하는 Google의 Gemini 2.5 Flash                    | gemini-2.5-flash           | ✅    |
| GM Gemini 3 Flash             | google-genai SDK를 통해 접근하는 Google의 Gemini 3 Flash 프리뷰               | gemini-3-flash-preview     | ✅    |
| GM Qwen 2.5 Coder 7B          | SFT + DPO + GRPO 파이프라인으로 파인튜닝된 오픈소스 모델                      | Qwen2.5-Coder-7B-Instruct | 🚧    |
| GM DeepSeek Coder V2 Lite     | SFT + DPO + GRPO 파이프라인으로 파인튜닝된 오픈소스 모델                      | DeepSeek-Coder-V2-Lite     | 🚧    |
| GM CodeLlama 7B               | SFT + DPO + GRPO 파이프라인으로 파인튜닝된 오픈소스 모델                      | CodeLlama-7b-Instruct      | 🚧    |

### 📡 새로운 모델

새로운 모델을 제안하고 싶다면, [저장소](https://github.com/360macky/generative-manim/issues)에 이슈를 열거나 [Discord 서버](https://discord.gg/SNdbPU2AMM)에서 대화해주세요.

## 🧠 훈련 파이프라인

GPT-4o에서 증류하는 3단계 파이프라인을 사용하여 Manim 코드를 생성하는 **오픈소스 모델**을 훈련하고 있습니다:

1. **SFT** (지도 파인튜닝): 5,000개 이상의 검증된 프롬프트→코드 쌍으로 훈련
2. **DPO** (직접 선호도 최적화): 렌더링 성공/실패 쌍에서 학습
3. **GRPO** (그룹 상대 정책 최적화): Manim 렌더러를 결정론적 보상 신호로 사용하는 RL

핵심 통찰: Manim은 **결정론적 검증기**입니다. 코드는 렌더링되거나 충돌합니다. 이는 DeepSeek-R1이 수학 답변 검사기를 사용하는 방식과 유사하게 보상 모델의 필요성을 대체합니다.

**기본 모델**: Qwen 2.5 Coder 7B, DeepSeek Coder V2 Lite, CodeLlama 7B. 모두 무료 Kaggle T4 GPU에 맞추기 위해 QLoRA(4-bit)를 사용합니다.

## 📏 벤치마크

Generative Manim은 이제 [`training/benchmarks`](./training/benchmarks)에서 전문가 Manim 코드 생성을 위한 실행 가능한 벤치마크 MVP를 포함합니다.

벤치마크는 프로그래밍 평가의 올바른 기본 요소를 기반으로 구축됩니다:

- 고정된 작업 모음
- 렌더링 기반 점수
- Manim 특화 구조 검사
- 확률적 코드 생성을 위한 pass@k
- 재현 가능한 JSONL 및 JSON 보고서

여기서 시작하세요:

```bash
cd training
python -m benchmarks.run export \
  --suite benchmarks/tasks/core_v1.jsonl \
  --output ./outputs/benchmarks/core_v1_prompts.jsonl
```

그런 다음 생성된 프롬프트 파일을 `python -m eval.generate_responses ...`와 함께 사용하거나, 다음 명령으로 전체 흐름을 실행하세요:

```bash
bash ./scripts/run_benchmark.sh qwen2.5-coder-7b ./outputs/grpo/qwen2.5-coder-7b benchmarks/tasks/core_v1.jsonl grpo 5 0.8 1,5
```

벤치마크 설계 및 워크플로우는 [`training/benchmarks/README.md`](./training/benchmarks/README.md)를 참조하세요.

여러 벤치마크 실행 결과가 있으면 다음으로 비교하세요:

```bash
cd training
python -m benchmarks.compare --results-dir ./outputs/benchmarks --suite core_v1
```

또는 매니페스트에서 전체 벤치마크 매트릭스를 실행하세요:

```bash
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/open_source_core_v1.json --dry-run
```

Featherless를 통해 호스팅된 오픈 웨이트 모델을 벤치마크할 수도 있습니다:

```bash
export FEATHERLESS_API_KEY="your-featherless-key"
cd training
python -m benchmarks.matrix --manifest benchmarks/manifests/featherless_core_v1.json --only qwen2.5-coder-7b-instruct-featherless
```

API 사용법, 스모크 테스트 및 전체 Featherless 벤치마크 워크플로우는 [`docs/featherless.md`](./docs/featherless.md)를 참조하세요.

## ✨ 스폰서

**Generative Manim**은 현재 **The Astronomical Software Company**의 후원을 받고 있습니다.

## 🙌 기여자

Generative Manim에 기여해주신 모든 분들께 감사드립니다!

| 기여자 | 기여 내용 |
| ------ | --------- |
| [@abdullahsohaill](https://github.com/abdullahsohaill) | google-genai를 통한 Gemini 2.5 Flash 및 Gemini 3 Flash 지원 추가 |
| [@tranquac](https://github.com/tranquac) | ffmpeg 비디오 내보내기의 명령 주입 취약점 수정 |
| [@Wing900](https://github.com/Wing900) | 중국어 README 번역 추가 |

## 🤲 기여하기

Generative Manim은 오픈소스 프로젝트입니다.

새로운 기능의 저자가 되거나, 버그를 수정하거나, 새로운 것을 기여하고 싶다면.

저장소를 포크하고 원하는 대로 변경하세요. [Pull requests](https://github.com/360macky/generative-manim/pulls)는 환영합니다. [Discord 서버](https://discord.gg/SNdbPU2AMM)에 참여하여 새로운 기능, 버그 또는 다른 주제에 대해 논의할 수도 있습니다.
