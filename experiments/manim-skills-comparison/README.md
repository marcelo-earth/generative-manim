# Manim Skills Comparison

Four Claude Code skills for generating Manim animations, each given the same prompt: a double pendulum chaos theory video.

Source: [r/manim - Best Manim skills comparison (with code + video)](https://www.reddit.com/r/manim/comments/1udub56/best_manim_skills_comparison_with_code_video/)

## Skills tested

| Folder | Skill | Approach |
|---|---|---|
| `sin-skill/` | None (base Manim) | Single class, manual `ValueTracker` + `updater` pattern. |
| `manim-composer/` | `manim-composer` | Plans a narrative arc (`scenes.md`) before writing any code. |
| `manim-skill-yusuke/` | [yusuke710/manim-skill](https://github.com/yusuke710/manim-skill) | One class per scene, native `.srt` subtitles via `add_subcaption()`, ffmpeg stitch. |
| `manim-video-affaan/` | [affaan-m/everything-claude-code (manim-video)](https://github.com/affaan-m/everything-claude-code) | Each scene must prove exactly one claim (visual thesis). |

All scripts simulate the double pendulum with the same RK4 integrator and the same physical parameters — the only variable is each skill's workflow.

## Requirements

```bash
pip install manim numpy
```

## Render

```bash
manim -pql sin-skill/double_pendulum.py DoublePendulumChaos
manim -pql manim-composer/double_pendulum_v2.py DoublePendulumNarrative
manim -pql manim-skill-yusuke/script.py Scene1_SinglePendulum  # and Scene2..Scene6
manim -pql manim-video-affaan/script.py Scene1_SameStart        # and Scene2..Scene3
```
