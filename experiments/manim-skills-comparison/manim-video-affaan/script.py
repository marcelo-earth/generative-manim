from manim import *
import numpy as np


# ── Physics ────────────────────────────────────────────────────────────────────

def _rk4(f, y, t, dt, *args):
    k1 = np.array(f(y, t, *args))
    k2 = np.array(f(y + dt/2*k1, t+dt/2, *args))
    k3 = np.array(f(y + dt/2*k2, t+dt/2, *args))
    k4 = np.array(f(y + dt*k3, t+dt, *args))
    return y + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def _ode(y, t, L1, L2, g=9.8):
    th1, w1, th2, w2 = y
    d = th2 - th1
    D1 = (2*L1 - L1*np.cos(d)**2)
    D2 = L2/L1*D1
    dw1 = (L1*w1**2*np.sin(d)*np.cos(d) + g*np.sin(th2)*np.cos(d)
           + L2*w2**2*np.sin(d) - 2*g*np.sin(th1)) / D1
    dw2 = (-L2*w2**2*np.sin(d)*np.cos(d) + 2*g*np.sin(th1)*np.cos(d)
           - 2*L1*w1**2*np.sin(d) - 2*g*np.sin(th2)) / D2
    return [w1, dw1, w2, dw2]

def sim(th1, th2, steps, dt=0.02, L1=1.5, L2=1.2):
    y = np.array([th1, 0.0, th2, 0.0])
    out = []
    for i in range(steps):
        x1 = L1*np.sin(y[0]); y1 = -L1*np.cos(y[0])
        x2 = x1+L2*np.sin(y[2]); y2 = y1-L2*np.cos(y[2])
        out.append((x1, y1, x2, y2))
        y = _rk4(_ode, y, i*dt, dt, L1, L2)
    return out


PIVOT = UP * 2.8
SC    = 1.0          # world → Manim scale
DT    = 0.02
TH0   = np.radians(120)

BLUE_PEND = "#4B9CD3"
RED_PEND  = "#E8474C"


def make_pend(pos, color):
    """Build rod1, rod2, bob1, bob2 at frame 0."""
    x1, y1, x2, y2 = pos[0]
    p1 = PIVOT + np.array([x1, y1, 0])*SC
    p2 = PIVOT + np.array([x2, y2, 0])*SC
    r1 = Line(PIVOT, p1, color=color, stroke_width=2.5, stroke_opacity=0.9)
    r2 = Line(p1, p2, color=color, stroke_width=2.5, stroke_opacity=0.9)
    b1 = Dot(p1, radius=0.11, color=color, fill_opacity=0.85)
    b2 = Dot(p2, radius=0.15, color=color)
    return r1, r2, b1, b2


def attach_updater(b2, pos, tracker, r1, r2, b1):
    def upd(mob, dt):
        i = min(int(tracker.get_value()), len(pos)-1)
        x1, y1, x2, y2 = pos[i]
        p1 = PIVOT + np.array([x1, y1, 0])*SC
        p2 = PIVOT + np.array([x2, y2, 0])*SC
        r1.put_start_and_end_on(PIVOT, p1)
        r2.put_start_and_end_on(p1, p2)
        b1.move_to(p1)
        mob.move_to(p2)
    b2.add_updater(upd)
    return upd


# ─────────────────────────────────────────────────────────────────────────────
# Scene 1 — Prove: they start identical
# Show both pendulums superimposed at the exact same position, then label the
# 0.5° gap so the viewer can see how small the difference actually is.
# ─────────────────────────────────────────────────────────────────────────────

class Scene1_SameStart(Scene):
    def construct(self):
        posA = sim(TH0, TH0, 1, DT)
        posB = sim(TH0 + np.radians(0.5), TH0, 1, DT)

        pivot = Dot(PIVOT, radius=0.07, color=WHITE)

        rA1, rA2, bA1, bA2 = make_pend(posA, BLUE_PEND)
        rB1, rB2, bB1, bB2 = make_pend(posB, RED_PEND)

        # Angle arc showing the 0.5° gap between the two first arms
        start_angle = np.radians(90) - TH0                # 90° - 120° = -30° (from East)
        delta_angle = np.radians(0.5)
        arc = Arc(
            radius=0.9,
            start_angle=start_angle,
            angle=delta_angle,
            color=YELLOW,
            stroke_width=3,
        ).shift(PIVOT)

        label_delta = MathTex(r"0.5^\circ", font_size=22, color=YELLOW)\
            .next_to(arc, LEFT, buff=0.1)

        thesis = Text(
            "Mismo inicio: 0.5° de diferencia",
            font_size=24, color=WHITE,
        ).to_edge(DOWN, buff=0.6)

        self.add(pivot)
        self.play(Create(rA1), Create(rA2), GrowFromCenter(bA1), GrowFromCenter(bA2), run_time=0.6)
        self.play(Create(rB1), Create(rB2), GrowFromCenter(bB1), GrowFromCenter(bB2), run_time=0.4)
        self.play(Create(arc), FadeIn(label_delta), FadeIn(thesis))
        self.wait(2)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 2 — Prove: they diverge
# Run both pendulums together. The first ~10s they track closely; then they
# explode apart. A vertical "sync" indicator fades out as they separate.
# ─────────────────────────────────────────────────────────────────────────────

class Scene2_Divergence(Scene):
    def construct(self):
        N = 2000
        posA = sim(TH0, TH0, N, DT)
        posB = sim(TH0 + np.radians(0.5), TH0, N, DT)

        pivot = Dot(PIVOT, radius=0.07, color=WHITE)
        rA1, rA2, bA1, bA2 = make_pend(posA, BLUE_PEND)
        rB1, rB2, bB1, bB2 = make_pend(posB, RED_PEND)
        trA = TracedPath(bA2.get_center, stroke_color=BLUE_PEND, stroke_width=1.5, dissipating_time=4)
        trB = TracedPath(bB2.get_center, stroke_color=RED_PEND,  stroke_width=1.5, dissipating_time=4)

        # Sync label (fades as they diverge — driven by angular distance)
        sync_label = Text("en sincronía", font_size=20, color=GREEN_B).to_edge(DOWN, buff=0.6)

        f = ValueTracker(0)
        attach_updater(bA2, posA, f, rA1, rA2, bA1)
        attach_updater(bB2, posB, f, rB1, rB2, bB1)

        # Compute frame index where angular distance first exceeds 45°
        diverge_frame = N - 1
        for i, (a, _, _, _) in enumerate(posA):
            diff = abs(posA[i][0] - posB[i][0])   # x1 difference as proxy
            if diff > 0.8:
                diverge_frame = i
                break

        self.add(pivot)
        self.play(
            Create(rA1), Create(rA2), GrowFromCenter(bA1), GrowFromCenter(bA2),
            Create(rB1), Create(rB2), GrowFromCenter(bB1), GrowFromCenter(bB2),
            FadeIn(sync_label),
            run_time=0.6,
        )
        self.add(trA, trB)

        # Phase 1: run to divergence frame
        t_diverge = diverge_frame * DT
        self.play(f.animate.set_value(diverge_frame), run_time=t_diverge, rate_func=linear)
        self.play(FadeOut(sync_label), run_time=0.5)

        # Phase 2: run the rest
        self.play(f.animate.set_value(N-1), run_time=(N-1-diverge_frame)*DT, rate_func=linear)

        bA2.clear_updaters()
        bB2.clear_updaters()
        self.wait(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Scene 3 — Prove: different outcomes, same physics
# Show the two full baked trails side-by-side with a minimal annotation.
# No rods, no motion — just the evidence.
# ─────────────────────────────────────────────────────────────────────────────

class Scene3_Outcomes(Scene):
    def construct(self):
        N = 2000
        posA = sim(TH0, TH0, N, DT)
        posB = sim(TH0 + np.radians(0.5), TH0, N, DT)

        def bake(pos, color):
            pts = [PIVOT + np.array([x2, y2, 0])*SC for _, _, x2, y2 in pos]
            m = VMobject(stroke_color=color, stroke_width=1.2, stroke_opacity=0.8)
            m.set_points_as_corners(pts)
            return m

        trailA = bake(posA, BLUE_PEND)
        trailB = bake(posB, RED_PEND)

        title = Text(
            "Misma física.\nResultados distintos.",
            font_size=28, color=WHITE, line_spacing=1.4,
        )
        sub = Text(
            "Determinismo no significa predictibilidad.",
            font_size=18, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.35)

        legend_a = Dot(color=BLUE_PEND, radius=0.09).to_edge(DOWN, buff=1.1).shift(LEFT*1.2)
        legend_b = Dot(color=RED_PEND,  radius=0.09).next_to(legend_a, RIGHT, buff=0.8)
        label_a  = Text("θ₁ = 120.0°", font_size=16, color=BLUE_PEND).next_to(legend_a, RIGHT, buff=0.12)
        label_b  = Text("θ₁ = 120.5°", font_size=16, color=RED_PEND).next_to(legend_b, RIGHT, buff=0.12)

        self.play(Create(trailA), Create(trailB), run_time=1.5)
        self.play(FadeIn(title, shift=UP*0.2))
        self.play(FadeIn(sub), FadeIn(legend_a), FadeIn(legend_b), FadeIn(label_a), FadeIn(label_b))
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
