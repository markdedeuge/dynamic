"""TDD tests for benchmark dynamical systems (Phase 2).

Tests written FIRST per TDD methodology. Systems are in:
- dynamic.systems.pl_map (2D PL map from Gardini et al.)
- dynamic.systems.duffing (Duffing oscillator)
- dynamic.systems.lorenz63 (Lorenz-63 system)
- dynamic.systems.oscillator (10D damped nonlinear oscillator)
- dynamic.systems.decision (Multistable decision-making model)
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 2D Piecewise-Linear Map (Gardini et al.)
# ---------------------------------------------------------------------------
class TestPLMap:
    """Tests for the 2D PL map from Appendix H.1."""

    @pytest.fixture()
    def map_fig3a_left(self):
        from dynamic.systems.pl_map import PLMap

        return PLMap.fig3a_left()

    @pytest.fixture()
    def map_fig5(self):
        from dynamic.systems.pl_map import PLMap

        return PLMap.fig5()

    def test_step_shape(self, map_fig3a_left):
        """Single step returns shape (2,)."""
        X = np.array([0.5, 0.3])
        out = map_fig3a_left.step(X)
        assert out.shape == (2,)

    def test_trajectory_shape(self, map_fig3a_left):
        """Trajectory has shape (T+1, 2)."""
        X0 = np.array([0.1, 0.2])
        traj = map_fig3a_left.trajectory(X0, T=100)
        assert traj.shape == (101, 2)

    def test_trajectory_first_state(self, map_fig3a_left):
        """First state in trajectory is X0."""
        X0 = np.array([0.1, 0.2])
        traj = map_fig3a_left.trajectory(X0, T=10)
        assert np.allclose(traj[0], X0)

    def test_piecewise_left_right(self, map_fig3a_left):
        """Map uses A_l for x <= 0, A_r for x >= 0."""
        X_left = np.array([-0.5, 0.3])
        X_right = np.array([0.5, 0.3])
        out_left = map_fig3a_left.step(X_left)
        out_right = map_fig3a_left.step(X_right)
        # Different matrices should give different results
        assert not np.allclose(out_left, out_right)

    def test_fixed_point_fig3a_left(self, map_fig3a_left):
        """Known fixed points satisfy F(z*) = z*."""
        fps = map_fig3a_left.analytical_fixed_points()
        assert len(fps) > 0
        for fp in fps:
            out = map_fig3a_left.step(fp)
            assert np.allclose(out, fp, atol=1e-10), (
                f"FP {fp} not fixed: F(z*)={out}"
            )

    def test_to_plrnn_params(self, map_fig3a_left):
        """PLRNN reformulation gives identical output to PL map."""
        params = map_fig3a_left.to_plrnn_params()
        assert "A" in params
        assert "W" in params
        assert "h" in params

        A = params["A"]
        W = params["W"]
        h = params["h"]

        # Test on both sides of the boundary
        for x_val in [-0.5, 0.5]:
            X = np.array([x_val, 0.3])
            plmap_out = map_fig3a_left.step(X)
            # PLRNN: z_t = (A + W D(z)) z + h
            D = np.diag((X > 0).astype(float))
            plrnn_out = (A + W @ D) @ X + h
            assert np.allclose(plmap_out, plrnn_out, atol=1e-10)

    def test_trajectory_bounded(self):
        """Trajectory stays bounded for map with stable attractor."""
        from dynamic.systems.pl_map import PLMap

        # fig3b_left has a stable fixed point + saddle cycle
        m = PLMap.fig3b_left()
        X0 = np.array([0.1, 0.2])
        traj = m.trajectory(X0, T=1000)
        assert np.all(np.isfinite(traj))
        assert np.max(np.abs(traj)) < 1e6

    def test_preset_configs_exist(self):
        """All preset configurations exist."""
        from dynamic.systems.pl_map import PLMap

        PLMap.fig3a_left()
        PLMap.fig3a_right()
        PLMap.fig3b_left()
        PLMap.fig3b_right()
        PLMap.fig5()

    def test_fig5_trajectory_bounded(self, map_fig5):
        """Fig 5 (chaotic) trajectory stays bounded."""
        X0 = np.array([0.1, 0.2])
        traj = map_fig5.trajectory(X0, T=5000)
        assert np.all(np.isfinite(traj))


# ---------------------------------------------------------------------------
# Duffing Oscillator
# ---------------------------------------------------------------------------
class TestDuffing:
    """Tests for the Duffing oscillator (Appendix H.3)."""

    def test_trajectory_shape(self):
        from dynamic.systems.duffing import generate_trajectory

        traj = generate_trajectory(x0=np.array([0.5, 0.0]), T=10.0, dt=0.01)
        assert traj.ndim == 2
        assert traj.shape[1] == 2  # (x, dx/dt)

    def test_two_equilibria(self):
        """Bistable Duffing has two stable equilibria."""
        from dynamic.systems.duffing import generate_trajectory

        # Start near one basin
        traj1 = generate_trajectory(x0=np.array([1.5, 0.0]), T=50.0, dt=0.01)
        # Start near other basin
        traj2 = generate_trajectory(x0=np.array([-1.5, 0.0]), T=50.0, dt=0.01)
        # Should converge to different equilibria
        end1 = traj1[-1, 0]
        end2 = traj2[-1, 0]
        assert end1 > 0 and end2 < 0, f"Expected opposite signs: {end1}, {end2}"

    def test_trajectories_bounded(self):
        """Trajectories stay bounded."""
        from dynamic.systems.duffing import generate_trajectory

        traj = generate_trajectory(x0=np.array([2.0, 1.0]), T=100.0, dt=0.01)
        assert np.all(np.isfinite(traj))
        assert np.max(np.abs(traj)) < 100

    def test_equilibria_locations(self):
        """Equilibria at x = ±sqrt(-α/β) = ±sqrt(10) ≈ ±3.16."""
        from dynamic.systems.duffing import generate_trajectory

        # α=-1, β=0.1 → x* = ±sqrt(1/0.1) = ±sqrt(10) ≈ ±3.162
        traj = generate_trajectory(x0=np.array([3.0, 0.0]), T=100.0, dt=0.01)
        x_eq = traj[-1, 0]
        assert abs(abs(x_eq) - np.sqrt(10)) < 0.1

    def test_ode_rhs(self):
        """ODE right-hand side has correct form."""
        from dynamic.systems.duffing import ode_rhs

        # At x=0, v=0: f = (0, -α·0 - β·0³ - δ·0) = (0, 0)
        state = np.array([0.0, 0.0])
        f = ode_rhs(0, state)
        assert np.allclose(f, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Lorenz-63
# ---------------------------------------------------------------------------
class TestLorenz63:
    """Tests for the Lorenz-63 system."""

    def test_trajectory_shape(self):
        from dynamic.systems.lorenz63 import generate_trajectory

        traj = generate_trajectory(
            x0=np.array([1.0, 1.0, 1.0]), T=10.0, dt=0.01
        )
        assert traj.ndim == 2
        assert traj.shape[1] == 3

    def test_three_fixed_points(self):
        """Lorenz-63 has 3 FPs: origin and ±(√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)."""
        from dynamic.systems.lorenz63 import fixed_points

        fps = fixed_points()
        assert len(fps) == 3
        # Origin
        assert any(np.allclose(fp, [0, 0, 0], atol=1e-10) for fp in fps)
        # Symmetric pair
        _sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        c = np.sqrt(beta * (rho - 1))
        fp_plus = np.array([c, c, rho - 1])
        fp_minus = np.array([-c, -c, rho - 1])
        assert any(np.allclose(fp, fp_plus, atol=1e-10) for fp in fps)
        assert any(np.allclose(fp, fp_minus, atol=1e-10) for fp in fps)

    def test_attractor_both_lobes(self):
        """Trajectory visits both wings of the butterfly attractor."""
        from dynamic.systems.lorenz63 import generate_trajectory

        traj = generate_trajectory(
            x0=np.array([1.0, 1.0, 1.0]), T=50.0, dt=0.01
        )
        x = traj[:, 0]
        # Should cross zero many times on the chaotic attractor
        sign_changes = np.sum(np.diff(np.sign(x)) != 0)
        assert sign_changes > 10, f"Expected many sign changes, got {sign_changes}"

    def test_trajectory_bounded(self):
        """Trajectory stays bounded on the attractor."""
        from dynamic.systems.lorenz63 import generate_trajectory

        traj = generate_trajectory(
            x0=np.array([1.0, 1.0, 1.0]), T=100.0, dt=0.01
        )
        assert np.all(np.isfinite(traj))
        assert np.max(np.abs(traj)) < 100

    def test_ode_rhs(self):
        """ODE RHS has correct form at origin."""
        from dynamic.systems.lorenz63 import ode_rhs

        f = ode_rhs(0, np.array([0.0, 0.0, 0.0]))
        assert np.allclose(f, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# 10D Damped Nonlinear Oscillator
# ---------------------------------------------------------------------------
class TestOscillator:
    """Tests for the 10D damped nonlinear oscillator."""

    def test_trajectory_shape(self):
        from dynamic.systems.oscillator import generate_trajectory

        x0 = np.random.default_rng(42).standard_normal(20)
        traj = generate_trajectory(x0=x0, T=10.0, dt=0.01)
        assert traj.ndim == 2
        assert traj.shape[1] == 20  # 10 position + 10 velocity

    def test_dimension(self):
        """System is 20D (10 positions + 10 velocities)."""
        from dynamic.systems.oscillator import ode_rhs

        state = np.zeros(20)
        f = ode_rhs(0, state)
        assert f.shape == (20,)

    def test_equilibrium_at_origin(self):
        """Origin is an equilibrium point."""
        from dynamic.systems.oscillator import ode_rhs

        state = np.zeros(20)
        f = ode_rhs(0, state)
        assert np.allclose(f, 0.0)

    def test_energy_decreasing(self):
        """Total energy decreases over time (damped system)."""
        from dynamic.systems.oscillator import generate_trajectory

        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(20) * 0.5
        traj = generate_trajectory(x0=x0, T=50.0, dt=0.01)

        # Kinetic energy: 0.5 * sum(v_i^2)
        velocities = traj[:, 10:]
        kinetic = 0.5 * np.sum(velocities**2, axis=1)
        # Total energy at start vs end should decrease
        # (with some damping, energy should dissipate)
        e_start = kinetic[100]  # skip transient
        e_end = kinetic[-1]
        assert e_end < e_start or e_end < 0.01  # should reach near-zero

    def test_trajectory_bounded(self):
        """Trajectory stays bounded."""
        from dynamic.systems.oscillator import generate_trajectory

        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(20) * 0.5
        traj = generate_trajectory(x0=x0, T=50.0, dt=0.01)
        assert np.all(np.isfinite(traj))


# ---------------------------------------------------------------------------
# Multistable Decision-Making Model
# ---------------------------------------------------------------------------
class TestDecision:
    """Tests for the multistable decision-making model."""

    def test_trajectory_shape(self):
        from dynamic.systems.decision import generate_trajectory

        x0 = np.array([0.0, 0.0, 0.0])
        traj = generate_trajectory(x0=x0, T=50.0, dt=0.01)
        assert traj.ndim == 2
        assert traj.shape[1] == 3  # h_E1, h_E2, h_inh

    def test_sigmoid_correct(self):
        """Sigmoid g_E(h; θ) = 1/(1 + exp(-(h - θ)))."""
        from dynamic.systems.decision import sigmoid

        theta = 5.0
        assert abs(sigmoid(5.0, theta) - 0.5) < 1e-10
        assert abs(sigmoid(100.0, theta) - 1.0) < 1e-5
        assert abs(sigmoid(-100.0, theta)) < 1e-5

    def test_two_stable_states(self):
        """System has two distinct stable steady states."""
        from dynamic.systems.decision import generate_trajectory

        # Start biased toward choice 1
        traj1 = generate_trajectory(
            x0=np.array([10.0, 0.0, 0.0]), T=100.0, dt=0.01
        )
        # Start biased toward choice 2
        traj2 = generate_trajectory(
            x0=np.array([0.0, 10.0, 0.0]), T=100.0, dt=0.01
        )
        end1 = traj1[-1]
        end2 = traj2[-1]
        # h_E1 should be high in one, h_E2 in the other
        assert end1[0] > end1[1], f"Choice 1 not dominant: {end1}"
        assert end2[1] > end2[0], f"Choice 2 not dominant: {end2}"
        # These should be distinct states
        assert not np.allclose(end1, end2, atol=0.5)

    def test_trajectory_bounded(self):
        """Trajectory stays bounded."""
        from dynamic.systems.decision import generate_trajectory

        x0 = np.array([5.0, 5.0, 5.0])
        traj = generate_trajectory(x0=x0, T=100.0, dt=0.01)
        assert np.all(np.isfinite(traj))
        assert np.max(np.abs(traj)) < 1000
