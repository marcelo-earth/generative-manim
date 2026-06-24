from manim import *
import numpy as np


# ─── Physics helpers ──────────────────────────────────────────────────────────

def pendulum_position(pivot, length, angle):
    """Return the (x,y,0) position of a pendulum bob given pivot, length, angle from vertical."""
    x = pivot[0] + length * np.sin(angle)
    y = pivot[1] - length * np.cos(angle)
    return np.array([x, y, 0])


def double_pendulum_derivatives(state, L1=1.5, L2=1.5, m1=1.0, m2=1.0, g=9.8):
    """Return dstate/dt for a double pendulum.
    state = [theta1, omega1, theta2, omega2]
    """
    t1, w1, t2, w2 = state
    dt = t2 - t1
    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(dt) ** 2
    denom2 = (L2 / L1) * denom1

    dt1 = w1
    dt2 = w2

    dw1 = (
        m2 * L1 * w1 ** 2 * np.sin(dt) * np.cos(dt)
        + m2 * g * np.sin(t2) * np.cos(dt)
        + m2 * L2 * w2 ** 2 * np.sin(dt)
        - (m1 + m2) * g * np.sin(t1)
    ) / denom1

    dw2 = (
        -m2 * L2 * w2 ** 2 * np.sin(dt) * np.cos(dt)
        + (m1 + m2) * g * np.sin(t1) * np.cos(dt)
        - (m1 + m2) * L1 * w1 ** 2 * np.sin(dt)
        - (m1 + m2) * g * np.sin(t2)
    ) / denom2

    return np.array([dt1, dw1, dt2, dw2])


def rk4_step(state, dt, **kwargs):
    """Single RK4 integration step."""
    k1 = double_pendulum_derivatives(state, **kwargs)
    k2 = double_pendulum_derivatives(state + 0.5 * dt * k1, **kwargs)
    k3 = double_pendulum_derivatives(state + 0.5 * dt * k2, **kwargs)
    k4 = double_pendulum_derivatives(state + dt * k3, **kwargs)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_double_pendulum(state0, n_steps, dt=0.02, **kwargs):
    """Pre-compute a trajectory. Returns array of shape (n_steps, 4)."""
    traj = np.zeros((n_steps, 4))
    traj[0] = state0
    for i in range(1, n_steps):
        traj[i] = rk4_step(traj[i - 1], dt, **kwargs)
    return traj


# ─── Scene 1: Single Pendulum ─────────────────────────────────────────────────

class Scene1_SinglePendulum(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        ## Scene1_SinglePendulum.title
        title = Text("The Predictable Pendulum", font_size=40, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.add_subcaption("A single pendulum is perfectly predictable.", duration=3)
        self.play(FadeIn(title), run_time=1.5)

        ## Scene1_SinglePendulum.pendulum_setup
        PIVOT = np.array([0.0, 1.8, 0.0])
        L = 2.2
        pivot_dot = Dot(point=PIVOT, radius=0.08, color=GREY_B)

        # ValueTracker for angle
        theta = ValueTracker(PI / 4)

        def make_rod():
            bob_pos = pendulum_position(PIVOT, L, theta.get_value())
            return Line(PIVOT, bob_pos, color=GREY_B, stroke_width=4)

        def make_bob():
            bob_pos = pendulum_position(PIVOT, L, theta.get_value())
            return Dot(point=bob_pos, radius=0.18, color=TEAL_C)

        rod = always_redraw(make_rod)
        bob = always_redraw(make_bob)

        self.add(pivot_dot, rod, bob)
        self.play(FadeIn(pivot_dot), FadeIn(rod), FadeIn(bob), run_time=0.8)

        ## Scene1_SinglePendulum.swing
        self.add_subcaption("Its motion repeats, tick by tick, forever.", duration=4)

        # Swing back and forth three times
        for target_angle in [PI / 4, -PI / 4, PI / 4, -PI / 4, PI / 4]:
            self.play(
                theta.animate.set_value(target_angle),
                run_time=1.2,
                rate_func=there_and_back if target_angle == PI / 4 else smooth,
            )

        self.wait(0.5)

        ## Scene1_SinglePendulum.label
        periodic_label = Text("Periodic. Predictable.", font_size=28, color=TEAL_A)
        periodic_label.next_to(title, DOWN, buff=0.3)
        self.add_subcaption("Periodic. Predictable. Boring.", duration=2)
        self.play(Write(periodic_label), run_time=1.2)
        self.wait(1)

        self.play(FadeOut(title), FadeOut(periodic_label), FadeOut(rod),
                  FadeOut(bob), FadeOut(pivot_dot), run_time=1.0)


# ─── Scene 2: Double Pendulum Intro ──────────────────────────────────────────

class Scene2_DoublePendulumIntro(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        ## Scene2_DoublePendulumIntro.title
        title = Text("Add a Second Pendulum", font_size=40, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.add_subcaption("Now attach a second pendulum below the first.", duration=3)
        self.play(FadeIn(title), run_time=1.0)

        ## Scene2_DoublePendulumIntro.setup
        PIVOT = np.array([0.0, 2.0, 0.0])
        L1, L2 = 1.6, 1.6
        # Start near vertical
        theta1_val = PI / 6
        theta2_val = PI / 4

        pivot_dot = Dot(point=PIVOT, radius=0.08, color=GREY_B)

        t1 = ValueTracker(theta1_val)
        t2 = ValueTracker(theta2_val)

        def joint_pos():
            return pendulum_position(PIVOT, L1, t1.get_value())

        def bob2_pos():
            jp = joint_pos()
            return pendulum_position(jp, L2, t2.get_value())

        rod1 = always_redraw(lambda: Line(PIVOT, joint_pos(), color=GREY_B, stroke_width=4))
        joint_dot = always_redraw(lambda: Dot(point=joint_pos(), radius=0.1, color=GREY_C))
        rod2 = always_redraw(lambda: Line(joint_pos(), bob2_pos(), color=GREY_B, stroke_width=4))
        bob2 = always_redraw(lambda: Dot(point=bob2_pos(), radius=0.18, color=GOLD))

        # First show just the first pendulum
        self.add(pivot_dot, rod1, joint_dot)
        self.play(Create(rod1), FadeIn(joint_dot), FadeIn(pivot_dot), run_time=1.0)

        # Then add second rod and bob
        self.add_subcaption("It still looks orderly... for a moment.", duration=3)
        self.play(Create(rod2), FadeIn(bob2), run_time=1.0)

        # Labels
        label1 = Text("rod 1", font_size=22, color=GREY_A)
        label2 = Text("rod 2", font_size=22, color=GREY_A)

        def update_label1(m):
            jp = joint_pos()
            midpoint = (PIVOT + jp) / 2
            m.move_to(midpoint + LEFT * 0.5)

        def update_label2(m):
            jp = joint_pos()
            bp = bob2_pos()
            midpoint = (jp + bp) / 2
            m.move_to(midpoint + LEFT * 0.5)

        label1.add_updater(update_label1)
        label2.add_updater(update_label2)
        self.add(label1, label2)
        self.play(FadeIn(label1), FadeIn(label2), run_time=0.6)
        self.wait(0.5)

        ## Scene2_DoublePendulumIntro.simulate_small
        # Simulate a small motion using pre-computed trajectory
        state0 = np.array([theta1_val, 0.0, theta2_val, 0.0])
        n = 120
        traj = simulate_double_pendulum(state0, n, dt=0.04)

        self.add_subcaption("The complexity grows with every swing.", duration=3)

        # Animate through pre-computed frames
        for i in range(0, n, 4):
            t1.set_value(traj[i, 0])
            t2.set_value(traj[i, 2])
            self.wait(0.04)

        self.wait(0.5)
        self.play(
            FadeOut(title), FadeOut(rod1), FadeOut(rod2), FadeOut(bob2),
            FadeOut(joint_dot), FadeOut(pivot_dot), FadeOut(label1), FadeOut(label2),
            run_time=1.0
        )


# ─── Scene 3: Chaos Reveal ────────────────────────────────────────────────────

class Scene3_ChaosReveal(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        ## Scene3_ChaosReveal.title
        title = Text("Chaos: Same Rules, Different Fate", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.add_subcaption(
            "Two double pendulums, nearly identical initial conditions.", duration=3
        )
        self.play(FadeIn(title), run_time=0.8)

        PIVOT = np.array([0.0, 2.5, 0.0])
        L1, L2 = 1.4, 1.4

        # Two initial conditions that differ by epsilon
        eps = 0.001
        state_A = np.array([PI / 2.1, 0.0, PI / 2.0, 0.0])
        state_B = state_A + np.array([eps, 0.0, 0.0, 0.0])

        N = 500
        dt = 0.02
        traj_A = simulate_double_pendulum(state_A, N, dt=dt)
        traj_B = simulate_double_pendulum(state_B, N, dt=dt)

        # Value trackers for frame index
        frame = ValueTracker(0)

        def get_idx():
            return int(frame.get_value())

        def joint_A():
            i = get_idx()
            return pendulum_position(PIVOT, L1, traj_A[i, 0])

        def bob2_A():
            jp = joint_A()
            return pendulum_position(jp, L2, traj_A[get_idx(), 2])

        def joint_B():
            i = get_idx()
            return pendulum_position(PIVOT, L1, traj_B[i, 0])

        def bob2_B():
            jp = joint_B()
            return pendulum_position(jp, L2, traj_B[get_idx(), 2])

        # Pendulum A (TEAL)
        pivot_dot = Dot(PIVOT, radius=0.08, color=GREY_B)
        rod1_A = always_redraw(lambda: Line(PIVOT, joint_A(), color=TEAL_C, stroke_width=3))
        joint_dot_A = always_redraw(lambda: Dot(joint_A(), radius=0.09, color=TEAL_C))
        rod2_A = always_redraw(lambda: Line(joint_A(), bob2_A(), color=TEAL_C, stroke_width=3))
        bob_A = always_redraw(lambda: Dot(bob2_A(), radius=0.16, color=TEAL_C,
                                          fill_opacity=1).set_stroke(TEAL_A, width=2))

        # Pendulum B (GOLD)
        rod1_B = always_redraw(lambda: Line(PIVOT, joint_B(), color=GOLD, stroke_width=3))
        joint_dot_B = always_redraw(lambda: Dot(joint_B(), radius=0.09, color=GOLD))
        rod2_B = always_redraw(lambda: Line(joint_B(), bob2_B(), color=GOLD, stroke_width=3))
        bob_B = always_redraw(lambda: Dot(bob2_B(), radius=0.16, color=GOLD,
                                          fill_opacity=1).set_stroke(GOLD_A, width=2))

        # Legends
        legend_A = VGroup(
            Dot(radius=0.12, color=TEAL_C),
            Text("Pendulum A", font_size=22, color=TEAL_C)
        ).arrange(RIGHT, buff=0.15).to_corner(DL, buff=0.5)
        legend_B = VGroup(
            Dot(radius=0.12, color=GOLD),
            Text("Pendulum B", font_size=22, color=GOLD)
        ).arrange(RIGHT, buff=0.15).next_to(legend_A, RIGHT, buff=0.6)

        diff_label = Text("Δθ₁ = 0.001 rad", font_size=20, color=RED_B)
        diff_label.to_corner(DR, buff=0.5)

        self.add(
            pivot_dot,
            rod1_A, joint_dot_A, rod2_A, bob_A,
            rod1_B, joint_dot_B, rod2_B, bob_B,
            legend_A, legend_B, diff_label
        )
        self.play(
            FadeIn(VGroup(rod1_A, joint_dot_A, rod2_A, bob_A,
                          rod1_B, joint_dot_B, rod2_B, bob_B, pivot_dot)),
            FadeIn(legend_A), FadeIn(legend_B), FadeIn(diff_label),
            run_time=1.0
        )

        ## Scene3_ChaosReveal.phase1: they start together
        self.add_subcaption("At first, they move together...", duration=3)
        self.play(frame.animate.set_value(80), run_time=3.0, rate_func=linear)

        ## Scene3_ChaosReveal.phase2: they diverge
        self.add_subcaption(
            "Then they diverge, completely, unpredictably.", duration=4
        )
        self.play(frame.animate.set_value(300), run_time=4.5, rate_func=linear)

        ## Scene3_ChaosReveal.chaos_label
        chaos_label = Text("This is CHAOS", font_size=44, color=RED_B)
        chaos_label.move_to(DOWN * 2.8)
        self.add_subcaption(
            "This is chaos: determinism without predictability.", duration=3
        )
        self.play(Write(chaos_label), run_time=1.2)
        self.wait(1.5)

        self.play(
            FadeOut(VGroup(title, rod1_A, joint_dot_A, rod2_A, bob_A,
                           rod1_B, joint_dot_B, rod2_B, bob_B, pivot_dot,
                           legend_A, legend_B, diff_label, chaos_label)),
            run_time=1.0
        )


# ─── Scene 4: Phase Portrait ──────────────────────────────────────────────────

class Scene4_PhasePortrait(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        ## Scene4_PhasePortrait.title
        title = Text("Phase Space: Where Chaos Lives", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.add_subcaption(
            "In phase space, orderly motion traces clean ellipses.", duration=3
        )
        self.play(FadeIn(title), run_time=0.8)

        # Build axes
        axes = Axes(
            x_range=[-PI, PI, PI / 2],
            y_range=[-10, 10, 5],
            x_length=9,
            y_length=5.5,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=False,
        )
        axes.move_to(ORIGIN + DOWN * 0.3)

        x_label = Text("θ₂ (angle)", font_size=22, color=GREY_A).next_to(
            axes.x_axis, DOWN, buff=0.25
        )
        y_label = Text("ω₂ (velocity)", font_size=22, color=GREY_A).next_to(
            axes.y_axis, LEFT, buff=0.15
        )

        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1.5)

        ## Scene4_PhasePortrait.simple_ellipse
        # Show a simple pendulum's phase portrait first (ellipse)
        ellipse_points = []
        for t_val in np.linspace(0, 2 * PI, 200):
            angle = 0.8 * np.cos(t_val)
            vel = -0.8 * np.sin(t_val) * 3
            pt = axes.c2p(angle, vel)
            ellipse_points.append(pt)

        ellipse = VMobject(color=TEAL_C, stroke_width=2.5)
        ellipse.set_points_as_corners(ellipse_points)
        ellipse.make_smooth()

        simple_label = Text("Simple pendulum: closed ellipse", font_size=20, color=TEAL_C)
        simple_label.to_corner(UL, buff=0.4).shift(DOWN * 1.0)

        self.add_subcaption("A simple pendulum traces a neat closed ellipse.", duration=2.5)
        self.play(Create(ellipse), FadeIn(simple_label), run_time=2.5)
        self.wait(0.5)

        ## Scene4_PhasePortrait.chaotic_trace
        # Simulate chaotic double pendulum and trace theta2 vs omega2
        state0 = np.array([PI / 2.1, 0.0, PI / 2.0, 0.0])
        N = 2000
        traj = simulate_double_pendulum(state0, N, dt=0.02)

        # Wrap theta2 to [-pi, pi]
        theta2_vals = traj[:, 2]
        omega2_vals = traj[:, 3]
        # Clip omega2 to axes range to avoid off-screen points
        omega2_vals = np.clip(omega2_vals, -9.5, 9.5)

        # Build phase path in chunks and draw it progressively
        self.play(FadeOut(ellipse), FadeOut(simple_label), run_time=0.8)

        chaos_label = Text("Double pendulum: never repeats", font_size=20, color=GOLD)
        chaos_label.to_corner(UL, buff=0.4).shift(DOWN * 1.0)
        self.add_subcaption(
            "But the double pendulum never repeats, it wanders forever.", duration=4
        )
        self.play(FadeIn(chaos_label), run_time=0.4)

        # Draw the phase portrait progressively as colored dots
        chunk_size = 100
        n_chunks = N // chunk_size
        colors = color_gradient([BLUE_B, PURPLE_B, RED_B], n_chunks)

        for c_idx in range(n_chunks):
            start = c_idx * chunk_size
            end = start + chunk_size
            pts = [
                axes.c2p(
                    np.clip((theta2_vals[k] + PI) % (2 * PI) - PI, -PI + 0.1, PI - 0.1),
                    omega2_vals[k]
                )
                for k in range(start, end)
            ]
            path = VMobject(
                color=colors[c_idx],
                stroke_width=1.2,
                stroke_opacity=0.8
            )
            path.set_points_as_corners(pts)
            self.add(path)
            self.wait(0.04)

        self.wait(1.0)

        self.play(
            FadeOut(VGroup(title, axes, x_label, y_label, chaos_label)),
            run_time=1.0,
        )


# ─── Scene 5: Lyapunov Exponent ───────────────────────────────────────────────

class Scene5_LyapunovExponent(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        ## Scene5_LyapunovExponent.title
        title = Text("Measuring Chaos: The Lyapunov Exponent", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.add_subcaption(
            "The distance between trajectories grows exponentially.", duration=3
        )
        self.play(FadeIn(title), run_time=0.8)

        ## Scene5_LyapunovExponent.divergence_plot
        # Simulate two trajectories and compute separation
        eps = 0.001
        state_A = np.array([PI / 2.1, 0.0, PI / 2.0, 0.0])
        state_B = state_A + np.array([eps, 0.0, 0.0, 0.0])
        N = 300
        dt = 0.02
        traj_A = simulate_double_pendulum(state_A, N, dt=dt)
        traj_B = simulate_double_pendulum(state_B, N, dt=dt)

        # Compute angular separation (simple distance in state space)
        sep = np.sqrt(
            (traj_A[:, 0] - traj_B[:, 0]) ** 2 + (traj_A[:, 2] - traj_B[:, 2]) ** 2
        )
        sep = np.clip(sep, 1e-12, None)
        log_sep = np.log(sep / eps)  # log of normalized separation

        # Axes for the divergence plot
        axes = Axes(
            x_range=[0, N * dt, 1.0],
            y_range=[-1, 8, 2],
            x_length=8.5,
            y_length=4.0,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=False,
        )
        axes.shift(DOWN * 0.5)
        x_label = Text("time (s)", font_size=20, color=GREY_A).next_to(
            axes.x_axis, DOWN, buff=0.2
        )
        y_label = Text("ln(separation)", font_size=20, color=GREY_A).next_to(
            axes.y_axis, LEFT, buff=0.15
        )

        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1.2)

        # Build separation curve
        time_vals = np.arange(N) * dt
        log_sep_clipped = np.clip(log_sep, -1, 7.5)

        sep_curve = axes.plot_parametric_curve(
            lambda t: np.array([
                t,
                float(np.interp(t, time_vals, log_sep_clipped)),
                0
            ]),
            t_range=[0, (N - 1) * dt, dt * 2],
            color=GOLD,
            stroke_width=3,
        )

        # Fitted exponential reference line
        lambda_fit = 2.5  # approximate Lyapunov exponent
        ref_line = axes.plot(
            lambda t: lambda_fit * t,
            x_range=[0, 2.5],
            color=RED_B,
            stroke_width=2,
        )

        self.add_subcaption(
            "The Lyapunov exponent λ > 0 is the signature of chaos.", duration=4
        )
        self.play(Create(sep_curve), run_time=3.0, rate_func=linear)
        self.play(Create(ref_line), run_time=1.0)

        # Lambda annotation
        lambda_eq = MathTex(r"\lambda > 0", font_size=44, color=RED_B)
        lambda_eq.to_corner(UR, buff=0.8).shift(DOWN * 1.5)
        self.play(Write(lambda_eq), run_time=0.8)

        forecast_text = Text(
            "Why forecasts fail beyond ~1 week", font_size=22, color=GREY_A
        )
        forecast_text.next_to(lambda_eq, DOWN, buff=0.3)
        self.add_subcaption(
            "It explains why long-range weather prediction is fundamentally impossible.",
            duration=3
        )
        self.play(FadeIn(forecast_text), run_time=0.8)
        self.wait(1.5)

        self.play(
            FadeOut(VGroup(title, axes, x_label, y_label, sep_curve,
                           ref_line, lambda_eq, forecast_text)),
            run_time=1.0
        )


# ─── Scene 6: Conclusion ──────────────────────────────────────────────────────

class Scene6_Conclusion(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        ## Scene6_Conclusion.split_screen
        # Left: simple pendulum trace (ellipse)
        # Right: chaotic trace

        left_axes = Axes(
            x_range=[-PI / 2, PI / 2, PI / 4],
            y_range=[-4, 4, 2],
            x_length=4.0,
            y_length=3.5,
            axis_config={"color": GREY_C, "stroke_width": 1.5},
            tips=False,
        ).shift(LEFT * 3.2 + DOWN * 0.5)

        right_axes = Axes(
            x_range=[-PI, PI, PI / 2],
            y_range=[-10, 10, 5],
            x_length=4.0,
            y_length=3.5,
            axis_config={"color": GREY_C, "stroke_width": 1.5},
            tips=False,
        ).shift(RIGHT * 3.2 + DOWN * 0.5)

        # Divider line
        divider = DashedLine(UP * 3.5, DOWN * 3.5, color=GREY_D, stroke_width=1.5)

        self.add_subcaption("The laws are exact. The future is determined.", duration=3)
        self.play(
            Create(left_axes), Create(right_axes),
            Create(divider), run_time=1.2
        )

        # Left ellipse (simple pendulum)
        simple_pts = []
        for t_val in np.linspace(0, 2 * PI, 300):
            angle = 0.6 * np.cos(t_val)
            vel = -0.6 * np.sin(t_val) * 3.5
            simple_pts.append(left_axes.c2p(angle, vel))
        simple_ellipse = VMobject(color=TEAL_C, stroke_width=2.5)
        simple_ellipse.set_points_as_corners(simple_pts)
        simple_ellipse.make_smooth()

        # Right chaotic trace (pre-computed)
        state0 = np.array([PI / 2.1, 0.0, PI / 2.0, 0.0])
        traj = simulate_double_pendulum(state0, 1500, dt=0.022)
        theta2_vals = np.clip(
            (traj[:, 2] + PI) % (2 * PI) - PI, -PI + 0.1, PI - 0.1
        )
        omega2_vals = np.clip(traj[:, 3], -9.5, 9.5)

        chaotic_pts = [right_axes.c2p(theta2_vals[k], omega2_vals[k])
                       for k in range(0, 1500, 3)]
        chaotic_path = VMobject(color=GOLD, stroke_width=1.0, stroke_opacity=0.7)
        chaotic_path.set_points_as_corners(chaotic_pts)

        # Labels above each panel
        label_simple = Text("Simple pendulum", font_size=22, color=TEAL_C)
        label_simple.next_to(left_axes, UP, buff=0.3)
        label_chaotic = Text("Double pendulum", font_size=22, color=GOLD)
        label_chaotic.next_to(right_axes, UP, buff=0.3)

        self.play(
            Create(simple_ellipse),
            Create(chaotic_path),
            FadeIn(label_simple),
            FadeIn(label_chaotic),
            run_time=2.5,
        )
        self.wait(0.5)

        ## Scene6_Conclusion.final_text
        self.add_subcaption(
            "But prediction? Impossible beyond a heartbeat.", duration=3
        )
        self.wait(2)

        final_text = Text(
            "Determinism  ≠  Predictability",
            font_size=42,
            color=WHITE,
        )
        final_text.set_color_by_gradient(TEAL_C, WHITE, GOLD)
        final_text.to_edge(DOWN, buff=0.6)

        self.add_subcaption(
            "Determinism does not mean predictability.", duration=3
        )
        self.play(Write(final_text), run_time=2.0)
        self.wait(2.5)

        self.play(FadeOut(Group(*self.mobjects)), run_time=1.5)
