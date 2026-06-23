from manim import *
import numpy as np


def double_pendulum_ode(state, t, L1, L2, m1, m2, g):
    th1, w1, th2, w2 = state
    d = th2 - th1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d) ** 2
    den2 = (L2 / L1) * den1

    dth1 = w1
    dw1 = (
        m2 * L1 * w1**2 * np.sin(d) * np.cos(d)
        + m2 * g * np.sin(th2) * np.cos(d)
        + m2 * L2 * w2**2 * np.sin(d)
        - (m1 + m2) * g * np.sin(th1)
    ) / den1

    dth2 = w2
    dw2 = (
        -m2 * L2 * w2**2 * np.sin(d) * np.cos(d)
        + (m1 + m2) * g * np.sin(th1) * np.cos(d)
        - (m1 + m2) * L1 * w1**2 * np.sin(d)
        - (m1 + m2) * g * np.sin(th2)
    ) / den2

    return dth1, dw1, dth2, dw2


def rk4_step(f, state, t, dt, *args):
    k1 = np.array(f(state, t, *args))
    k2 = np.array(f(state + dt / 2 * k1, t + dt / 2, *args))
    k3 = np.array(f(state + dt / 2 * k2, t + dt / 2, *args))
    k4 = np.array(f(state + dt * k3, t + dt, *args))
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(th1_0, th2_0, n_steps, dt, L1=1.5, L2=1.2, m1=1.0, m2=1.0, g=9.8):
    state = np.array([th1_0, 0.0, th2_0, 0.0])
    positions = []
    for i in range(n_steps):
        x1 = L1 * np.sin(state[0])
        y1 = -L1 * np.cos(state[0])
        x2 = x1 + L2 * np.sin(state[2])
        y2 = y1 - L2 * np.cos(state[2])
        positions.append((x1, y1, x2, y2))
        state = rk4_step(double_pendulum_ode, state, i * dt, dt, L1, L2, m1, m2, g)
    return positions


class DoublePendulumChaos(Scene):
    def construct(self):
        # --- parameters ---
        L1, L2 = 1.5, 1.2
        SCALE = 1.4          # world-units to Manim units
        PIVOT = UP * 2.8
        DT = 0.016
        N_STEPS = 520        # ~8 s at 60 fps equivalent
        TRAIL_LEN = 120

        # Two pendulums with a tiny angle difference (chaos demo)
        CONFIGS = [
            {"th1": np.radians(120), "th2": np.radians(120), "color": BLUE},
            {"th1": np.radians(120.5), "th2": np.radians(120), "color": RED},
        ]

        # Pre-simulate
        all_pos = []
        for cfg in CONFIGS:
            all_pos.append(simulate(cfg["th1"], cfg["th2"], N_STEPS, DT, L1, L2))

        # --- title ---
        title = Text("Péndulo doble", font_size=36, color=WHITE).to_edge(UP)
        subtitle = Text(
            "Delta theta = 0.5° de diferencia",
            font_size=22,
            color=GRAY_B,
        ).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN * 0.3), FadeIn(subtitle, shift=DOWN * 0.3))
        self.wait(0.4)

        # --- static pivot dot ---
        pivot_dot = Dot(PIVOT, radius=0.08, color=WHITE)
        self.add(pivot_dot)

        # Build Manim objects per pendulum
        rods = []
        bobs = []
        trails = []
        trail_points = [[] for _ in CONFIGS]

        for i, cfg in enumerate(CONFIGS):
            x1, y1, x2, y2 = all_pos[i][0]
            p1 = PIVOT + np.array([x1, y1, 0]) * SCALE
            p2 = PIVOT + np.array([x2, y2, 0]) * SCALE

            rod1 = Line(PIVOT, p1, color=GRAY_A, stroke_width=2.5)
            rod2 = Line(p1, p2, color=GRAY_A, stroke_width=2.5)
            bob1 = Dot(p1, radius=0.12, color=cfg["color"]).set_opacity(0.8)
            bob2 = Dot(p2, radius=0.16, color=cfg["color"])
            trail = VMobject(stroke_width=1.5, stroke_color=cfg["color"])
            trail.set_points_as_corners([p2, p2])

            rods.append((rod1, rod2))
            bobs.append((bob1, bob2))
            trails.append(trail)
            trail_points[i].append(p2.copy())

            self.add(rod1, rod2, bob1, bob2, trail)

        # --- animate ---
        def update_pendulum(pend_idx, frame):
            x1, y1, x2, y2 = all_pos[pend_idx][frame]
            p1 = PIVOT + np.array([x1, y1, 0]) * SCALE
            p2 = PIVOT + np.array([x2, y2, 0]) * SCALE

            rod1, rod2 = rods[pend_idx]
            bob1, bob2 = bobs[pend_idx]
            trail = trails[pend_idx]

            rod1.put_start_and_end_on(PIVOT, p1)
            rod2.put_start_and_end_on(p1, p2)
            bob1.move_to(p1)
            bob2.move_to(p2)

            trail_points[pend_idx].append(p2.copy())
            if len(trail_points[pend_idx]) > TRAIL_LEN:
                trail_points[pend_idx].pop(0)

            pts = trail_points[pend_idx]
            if len(pts) >= 2:
                trail.set_points_as_corners(pts)

        frame = ValueTracker(0)

        def make_updater(idx):
            def updater(mob, dt):
                f = int(frame.get_value())
                if f < N_STEPS:
                    update_pendulum(idx, f)
            return updater

        for idx in range(len(CONFIGS)):
            # attach updater to the bob (dummy target just to hook into the scene)
            bobs[idx][1].add_updater(make_updater(idx))

        self.play(
            frame.animate.set_value(N_STEPS - 1),
            run_time=N_STEPS * DT,
            rate_func=linear,
        )

        for idx in range(len(CONFIGS)):
            bobs[idx][1].remove_updater(make_updater(idx))

        # --- end card ---
        end_text = Text(
            "Mismas condiciones iniciales,\ntrayectorias completamente distintas.",
            font_size=24,
            color=WHITE,
            line_spacing=1.3,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(end_text, shift=UP * 0.2))
        self.wait(2.5)
        self.play(FadeOut(end_text))
