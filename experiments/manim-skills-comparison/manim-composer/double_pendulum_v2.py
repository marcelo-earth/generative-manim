from manim import *
import numpy as np


# ── Physics ────────────────────────────────────────────────────────────────────

def rk4(f, y, t, dt, *args):
    k1 = np.array(f(y, t, *args))
    k2 = np.array(f(y + dt/2*k1, t+dt/2, *args))
    k3 = np.array(f(y + dt/2*k2, t+dt/2, *args))
    k4 = np.array(f(y + dt*k3,   t+dt,   *args))
    return y + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def ode_simple(y, t, L, g):
    return [y[1], -g/L*np.sin(y[0])]


def ode_double(y, t, L1, L2, m1, m2, g):
    th1, w1, th2, w2 = y
    d = th2 - th1
    D1 = (m1+m2)*L1 - m2*L1*np.cos(d)**2
    D2 = L2/L1*D1
    dw1 = (m2*L1*w1**2*np.sin(d)*np.cos(d) + m2*g*np.sin(th2)*np.cos(d)
           + m2*L2*w2**2*np.sin(d) - (m1+m2)*g*np.sin(th1)) / D1
    dw2 = (-m2*L2*w2**2*np.sin(d)*np.cos(d) + (m1+m2)*g*np.sin(th1)*np.cos(d)
           - (m1+m2)*L1*w1**2*np.sin(d) - (m1+m2)*g*np.sin(th2)) / D2
    return [w1, dw1, w2, dw2]


def sim_simple(th0, steps, dt, L=2.0, g=9.8):
    y = np.array([th0, 0.0])
    out = []
    for i in range(steps):
        out.append((L*np.sin(y[0]), -L*np.cos(y[0])))
        y = rk4(ode_simple, y, i*dt, dt, L, g)
    return out


def sim_double(th1, th2, steps, dt, L1=1.5, L2=1.2, g=9.8):
    y = np.array([th1, 0.0, th2, 0.0])
    out = []
    for i in range(steps):
        x1 = L1*np.sin(y[0]); y1 = -L1*np.cos(y[0])
        x2 = x1+L2*np.sin(y[2]); y2 = y1-L2*np.cos(y[2])
        out.append((x1, y1, x2, y2))
        y = rk4(ode_double, y, i*dt, dt, L1, L2, 1.0, 1.0, g)
    return out


# ── Scene ──────────────────────────────────────────────────────────────────────

class DoublePendulumNarrative(Scene):
    def construct(self):
        SCALE  = 1.1
        PIVOT  = UP * 2.6
        DT     = 0.02
        L1, L2 = 1.5, 1.2

        # Pre-simulate all trajectories up front
        S1  = sim_simple(np.radians(40), 750, DT)
        S3  = sim_double(np.radians(120), np.radians(120), 750, DT, L1, L2)
        th0 = np.radians(120)
        S4A = sim_double(th0,                    th0, 2000, DT, L1, L2)
        S4B = sim_double(th0 + np.radians(0.5),  th0, 2000, DT, L1, L2)

        pivot_dot = Dot(PIVOT, radius=0.07, color=WHITE)
        self.add(pivot_dot)

        # ── Scene 1: Predictable single pendulum ───────────────────────────────
        x0, y0 = S1[0]
        bob_s = Dot(PIVOT + np.array([x0, y0, 0])*SCALE, radius=0.16, color=BLUE_B)
        rod_s = Line(PIVOT, bob_s.get_center(), color=GRAY_A, stroke_width=3)

        f1 = ValueTracker(0)

        def upd_bob_s(mob, dt):
            f = min(int(f1.get_value()), len(S1)-1)
            x, y = S1[f]
            mob.move_to(PIVOT + np.array([x, y, 0])*SCALE)

        def upd_rod_s(mob, dt):
            mob.put_start_and_end_on(PIVOT, bob_s.get_center())

        bob_s.add_updater(upd_bob_s)
        rod_s.add_updater(upd_rod_s)

        trace_s = TracedPath(
            bob_s.get_center,
            stroke_color=BLUE_B, stroke_width=1.8, dissipating_time=3,
        )
        label_pred = Text("Predecible", font_size=26, color=GRAY_A).to_edge(DOWN, buff=0.7)

        self.play(Create(rod_s), GrowFromCenter(bob_s), run_time=0.5)
        self.add(trace_s)
        self.play(FadeIn(label_pred))
        self.play(f1.animate.set_value(len(S1)-1), run_time=len(S1)*DT, rate_func=linear)

        bob_s.remove_updater(upd_bob_s)
        rod_s.remove_updater(upd_rod_s)

        # ── Scene 2: Add a second arm ──────────────────────────────────────────
        xe, ye = S1[-1]
        p1e = PIVOT + np.array([xe, ye, 0])*SCALE
        p2e = p1e + DOWN * L2 * SCALE

        rod2_appear = Line(p1e, p2e, color=GRAY_B, stroke_width=3)
        bob2_appear = Dot(p2e, radius=0.14, color=BLUE_B)
        label_q = Text("?", font_size=40, color=YELLOW).move_to(label_pred)

        self.play(FadeOut(trace_s), FadeOut(label_pred))
        self.play(Create(rod2_appear), GrowFromCenter(bob2_appear), run_time=0.8)
        self.play(FadeIn(label_q))
        self.wait(0.6)

        # ── Scene 3: Single double pendulum, chaotic trail ─────────────────────
        self.play(
            FadeOut(rod_s), FadeOut(bob_s),
            FadeOut(rod2_appear), FadeOut(bob2_appear),
            FadeOut(label_q),
        )

        x1_0, y1_0, x2_0, y2_0 = S3[0]
        rod1_d = Line(PIVOT,
                      PIVOT + np.array([x1_0, y1_0, 0])*SCALE,
                      color=GRAY_A, stroke_width=3)
        rod2_d = Line(PIVOT + np.array([x1_0, y1_0, 0])*SCALE,
                      PIVOT + np.array([x2_0, y2_0, 0])*SCALE,
                      color=GRAY_A, stroke_width=3)
        bob1_d = Dot(PIVOT + np.array([x1_0, y1_0, 0])*SCALE, radius=0.13, color=BLUE_C)
        bob2_d = Dot(PIVOT + np.array([x2_0, y2_0, 0])*SCALE, radius=0.16, color=YELLOW)
        trace_d = TracedPath(
            bob2_d.get_center,
            stroke_color=YELLOW, stroke_width=2, dissipating_time=4,
        )

        f3 = ValueTracker(0)

        def upd_dp(mob, dt):
            f = min(int(f3.get_value()), len(S3)-1)
            x1, y1, x2, y2 = S3[f]
            p1 = PIVOT + np.array([x1, y1, 0])*SCALE
            p2 = PIVOT + np.array([x2, y2, 0])*SCALE
            rod1_d.put_start_and_end_on(PIVOT, p1)
            rod2_d.put_start_and_end_on(p1, p2)
            bob1_d.move_to(p1)
            mob.move_to(p2)

        bob2_d.add_updater(upd_dp)

        self.play(
            Create(rod1_d), Create(rod2_d),
            GrowFromCenter(bob1_d), GrowFromCenter(bob2_d),
            run_time=0.6,
        )
        self.add(trace_d)
        self.play(f3.animate.set_value(len(S3)-1), run_time=len(S3)*DT, rate_func=linear)

        bob2_d.remove_updater(upd_dp)

        # ── Scene 4: Two pendulums, 0.5 degree apart ───────────────────────────
        self.play(
            FadeOut(rod1_d), FadeOut(rod2_d),
            FadeOut(bob1_d), FadeOut(bob2_d),
            FadeOut(trace_d),
        )

        def build_pend(pos, col):
            x1, y1, x2, y2 = pos[0]
            p1 = PIVOT + np.array([x1, y1, 0])*SCALE
            p2 = PIVOT + np.array([x2, y2, 0])*SCALE
            r1 = Line(PIVOT, p1, color=col, stroke_width=2.5)
            r2 = Line(p1, p2, color=col, stroke_width=2.5)
            b1 = Dot(p1, radius=0.12, color=col)
            b2 = Dot(p2, radius=0.15, color=col)
            tr = TracedPath(b2.get_center, stroke_color=col, stroke_width=1.8, dissipating_time=5)
            return r1, r2, b1, b2, tr

        rA1, rA2, bA1, bA2, trA = build_pend(S4A, BLUE)
        rB1, rB2, bB1, bB2, trB = build_pend(S4B, RED)

        delta_lbl = MathTex(r"\Delta\theta = 0.5^\circ", font_size=26, color=GRAY_A)\
            .to_corner(UR, buff=0.5)

        f4 = ValueTracker(0)

        def make_upd(pos, r1, r2, b1):
            def upd(mob, dt):
                f = min(int(f4.get_value()), len(pos)-1)
                x1, y1, x2, y2 = pos[f]
                p1 = PIVOT + np.array([x1, y1, 0])*SCALE
                p2 = PIVOT + np.array([x2, y2, 0])*SCALE
                r1.put_start_and_end_on(PIVOT, p1)
                r2.put_start_and_end_on(p1, p2)
                b1.move_to(p1)
                mob.move_to(p2)
            return upd

        bA2.add_updater(make_upd(S4A, rA1, rA2, bA1))
        bB2.add_updater(make_upd(S4B, rB1, rB2, bB1))

        self.play(
            Create(rA1), Create(rA2), GrowFromCenter(bA1), GrowFromCenter(bA2),
            Create(rB1), Create(rB2), GrowFromCenter(bB1), GrowFromCenter(bB2),
            FadeIn(delta_lbl),
            run_time=0.7,
        )
        self.add(trA, trB)
        self.play(f4.animate.set_value(len(S4A)-1), run_time=len(S4A)*DT, rate_func=linear)

        bA2.clear_updaters()
        bB2.clear_updaters()

        # ── Scene 5: End card ──────────────────────────────────────────────────
        self.play(
            FadeOut(rA1), FadeOut(rA2), FadeOut(bA1), FadeOut(bA2),
            FadeOut(rB1), FadeOut(rB2), FadeOut(bB1), FadeOut(bB2),
            FadeOut(delta_lbl), FadeOut(pivot_dot),
        )

        end_title = Text(
            "Dependencia sensible\nde las condiciones iniciales",
            font_size=30, color=WHITE, line_spacing=1.3,
        )
        end_sub = Text(
            "El efecto mariposa",
            font_size=22, color=GRAY_B,
        ).next_to(end_title, DOWN, buff=0.4)

        self.play(FadeIn(end_title, shift=UP*0.2))
        self.play(FadeIn(end_sub, shift=UP*0.2))
        self.wait(3)
        self.play(
            FadeOut(end_title), FadeOut(end_sub),
            FadeOut(trA), FadeOut(trB),
        )
