# Comparación de skills de Manim para Claude Code

Cuatro skills de Claude Code para generar animaciones con Manim, puestas a prueba con el mismo encargo: un video sobre el péndulo doble como ejemplo de teoría del caos.

Origen de esta comparación: [r/manim - Best Manim skills comparison (with code + video)](https://www.reddit.com/r/manim/comments/1udub56/best_manim_skills_comparison_with_code_video/)

## Skills comparados

| Carpeta | Skill | Enfoque |
|---|---|---|
| `sin-skill/` | Ninguno (Manim base) | Una sola clase, patrón manual de `ValueTracker` + `updater`. |
| `manim-composer/` | `manim-composer` | Planifica un arco narrativo (`scenes.md`) antes de escribir código. |
| `manim-skill-yusuke/` | [yusuke710/manim-skill](https://github.com/yusuke710/manim-skill) | Una clase por escena, subtítulos nativos vía `add_subcaption()`, stitching con ffmpeg. |
| `manim-video-affaan/` | [affaan-m/everything-claude-code (manim-video)](https://github.com/affaan-m/everything-claude-code) | Cada escena debe demostrar una sola idea (tesis visual). |

Todos los scripts simulan el péndulo doble con el mismo integrador RK4 y los mismos parámetros físicos, para que la única variable real sea el flujo de trabajo de cada skill.

Los videos resultantes y el análisis comparativo completo no se incluyen en este repositorio; este folder solo contiene el código fuente de las escenas (`Scene` classes de Manim) de cada skill.

## Requisitos

```bash
pip install manim numpy
```

## Render

```bash
manim -pql sin-skill/double_pendulum.py DoublePendulumChaos
manim -pql manim-composer/double_pendulum_v2.py DoublePendulumNarrative
manim -pql manim-skill-yusuke/script.py Scene1_SinglePendulum  # y las demás Scene1-6
manim -pql manim-video-affaan/script.py Scene1_SameStart        # y las demás Scene1-3
```
