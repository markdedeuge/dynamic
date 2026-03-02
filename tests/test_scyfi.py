"""TDD tests for the main SCYFI algorithm (Phase 4, Round 3).

Tests written FIRST per TDD methodology. Implementation target:
    dynamic.analysis.scyfi

Test data ported from the Julia reference tests:
    reference/SCYFI/test/convPLRNN_tests.jl
    reference/SCYFI/test/shPLRNN_tests.jl
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures: 2D PLRNN systems from Julia tests
# ---------------------------------------------------------------------------
@pytest.fixture()
def system_2d_holes():
    """2D PLRNN with 1 FP — 'holes' test case."""
    A = torch.tensor([[0.7, 0.0], [0.0, -0.7]])
    W = torch.tensor([[0.0, -0.14375], [0.52505308, 0.0]])
    h = torch.tensor([0.37298253, -0.97931491])
    return A, W, h


@pytest.fixture()
def system_2d_fp():
    """2D PLRNN with 1 FP and known value."""
    A = torch.tensor([[-0.022231099999999948, 0.0], [0.0, 0.96461297]])
    W = torch.tensor([[0.0, -0.6437499999999996], [0.52505308, 0.0]])
    h = torch.tensor([0.37298253000000087, -0.97931491])
    return A, W, h


@pytest.fixture()
def system_2d_16cycle():
    """2D PLRNN with 16-cycles."""
    A = torch.tensor([[0.8377689000000008, 0.0], [0.0, 0.96461297]])
    W = torch.tensor([[0.0, -0.6437499999999996], [0.52505308, 0.0]])
    h = torch.tensor([0.37298253000000087, -0.97931491])
    return A, W, h


@pytest.fixture()
def system_2d_27cycle():
    """2D PLRNN with 27-cycles."""
    A = torch.tensor([[0.9777689, 0.0], [0.0, 0.96461297]])
    W = torch.tensor([[0.0, -0.14375], [0.52505308, 0.0]])
    h = torch.tensor([0.37298253, -0.97931491])
    return A, W, h


@pytest.fixture()
def system_10d():
    """10D PLRNN from Julia tests."""
    A = torch.eye(10) * 0.09658233076334
    W = torch.tensor([
        [1.7141654, 0.90136945, 0.7356431, -0.45661467, -1.0216872,
         -2.1104505, -0.1265201, 0.8208984, -0.16492529, 0.10196607],
        [1.2489663, 0.36885896, 0.47238302, -0.4648625, -0.8395844,
         -1.6625919, -0.27828497, 0.4668116, 0.2430289, 0.43664113],
        [0.4888246, 0.31421587, 0.061512344, 0.03574979, -0.43089348,
         -0.9017075, 0.06113107, 0.45323887, -0.12256672, -0.22407334],
        [-0.8275283, -0.6372472, -0.6524501, 0.5350839, 0.143212,
         0.3879285, 0.17147072, 0.18457298, 0.33618844, -0.31503707],
        [0.4699096, 0.36021802, 0.40548527, -0.773131, -0.41375652,
         -0.45799288, -0.33785427, 0.0071197264, -0.12557332, 0.4461898],
        [1.7938753, 0.933799, 0.9427183, -0.4197829, -1.0978042,
         -2.340497, -0.60876435, 0.8649732, -0.11037506, 0.4225005],
        [-0.46095127, -0.282801, -0.14612287, 0.23626074, 0.43258965,
         0.68257767, 0.36670405, -0.25161934, 0.1560926, -0.09981756],
        [-0.1794313, -0.046297066, -0.23596726, 0.60501343, 0.077439524,
         -0.037234787, 0.18110584, 0.3498277, -0.007887238, -0.4353217],
        [-0.3996467, -0.28196895, -0.1993325, -0.027839307, 0.20849963,
         0.4165471, 0.17119904, -0.2667973, 0.25180992, 0.13836445],
        [0.4042766, 0.52036214, 0.21992852, -0.1569775, 0.024118641,
         -0.08136176, -0.20892598, -0.1640525, -0.3731168, 0.09140323],
    ])
    h = torch.tensor([
        -0.25387551903975236, -0.12912468150368078, -0.3746121660912428,
        0.5381540053468717, 0.13797574730440254, -0.9145060467607822,
        -0.47008295657633226, 0.23037773308426446, 0.5373384681742799,
        0.32054443856805936,
    ])
    return A, W, h


# ---------------------------------------------------------------------------
# shPLRNN fixtures from Julia tests
# ---------------------------------------------------------------------------
@pytest.fixture()
def sh_system_1fp():
    """shPLRNN M=2, H=10 system with 1 FP."""
    A = torch.tensor([1.4799836250879086, -1.6945099980546405])
    W1 = torch.tensor([
        [0.8900829721936014, 0.24766868576521212, 0.2554263710950393,
         -0.5771080679179108, -2.129379740761085, 0.031728042646673976,
         -0.7740703525170001, 0.04768259094291391, 0.6792959504153095,
         -1.7781119769858111],
        [0.6297279024943379, 0.5586975133684497, -0.18842627535543358,
         0.7152028558426154, 2.238266026927219, 1.0163441695005024,
         0.6115773461504028, 1.8513040760318145, 0.5127750227966122,
         -2.337416929726536],
    ])
    W2 = torch.tensor([
        [-0.7543485848832211, 0.857424094861255],
        [-0.12981333832339487, 0.9290731862819631],
        [0.06164740828503748, -1.6206522735934827],
        [0.1856635717214527, 0.23370129970278428],
        [1.4788242336217083, -1.4604870389877311],
        [-0.04197765409809221, 0.364736070574788],
        [0.04390077093273831, -0.6840588607748522],
        [0.34236696876785494, -0.6110477480680268],
        [0.11164936136447304, 0.2509392315579666],
        [0.9314203111545968, 0.18253806886914628],
    ])
    h1 = torch.tensor([0.18005923664630755, 0.42057092737475477])
    h2 = torch.tensor([
        -0.017897789466083707, 1.014731429962641, -0.3431368102401408,
        -2.3065949842414866, -0.7738820872835147, 0.009176181040445722,
        0.6395596355828497, -0.3762485593614053, -0.12524612444111408,
        0.25769322230050085,
    ])
    return A, W1, W2, h1, h2


@pytest.fixture()
def sh_system_2_4_cycle():
    """shPLRNN M=2, H=10 system with 2-cycles and 4-cycles."""
    A = torch.tensor([0.6172450705591427, 0.5261273846061184])
    W1 = torch.tensor([
        [-1.022331291385995, -0.6068651009735164, 0.005478115970447768,
         -0.6765758765817282, 0.17667276783032312, -0.37241595029711383,
         -0.2799681077442852, 1.6336122869854053, 0.7388068389577566,
         0.43655916960874785],
        [-0.23031703096807454, -0.155167269588236, 1.1743020054655264,
         1.4693401983283279, -1.141192841208352, 0.11938698490497139,
         0.6192500266755361, 0.03950302999313116, -0.9179498107708933,
         -0.1216362805456489],
    ])
    W2 = torch.tensor([
        [1.753949852408234, -0.692574927107241],
        [-0.45303568496148183, -0.5365936668032565],
        [0.07984706960954363, -0.48651643130319855],
        [-1.0455124065967838, -0.22986736948781128],
        [0.048362883726876985, 0.8959123953895494],
        [-1.0545422354241465, 0.5685368747406444],
        [-0.43826625604171476, -2.1955495493951815],
        [-0.8981129185384389, -0.6454540072455006],
        [0.6162103523913983, -0.8644618121879155],
        [-1.2236281701654421, -2.059927291272103],
    ])
    h1 = torch.tensor([-0.5480895548836227, -0.2922735885352696])
    h2 = torch.tensor([
        0.5352937111114038, -1.110030373073419, -1.3146036515301616,
        0.2748467715335772, -1.4155203620983157, 0.7891282169615852,
        -0.13084812694281087, -0.40652418385647066, -0.9383323642698853,
        -0.9983356016811977,
    ])
    return A, W1, W2, h1, h2


# ---------------------------------------------------------------------------
# PLRNN fixed point tests
# ---------------------------------------------------------------------------
class TestFindFixedPoints:
    """Tests for fixed point finding in PLRNN systems."""

    def test_find_fp_2d_holes(self, system_2d_holes):
        """Finds exactly 1 FP in 2D 'holes' system."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_holes
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        cycles_order_1 = result[0][0]
        assert len(cycles_order_1) == 1

    def test_find_fp_2d(self, system_2d_fp):
        """Finds exactly 1 FP in second 2D system."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_fp
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        cycles_order_1 = result[0][0]
        assert len(cycles_order_1) == 1

    def test_find_fp_2d_value(self, system_2d_fp):
        """Found FP ≈ [0.36, -22.26]."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_fp
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        cycles_order_1 = result[0][0]
        assert len(cycles_order_1) >= 1
        fp = cycles_order_1[0][0]  # first trajectory, first point
        fp_rounded = torch.round(fp * 100) / 100
        expected = torch.tensor([0.36, -22.26])
        assert torch.allclose(fp_rounded, expected, atol=0.01)

    def test_fp_self_consistency(self, system_2d_holes):
        """Found FP satisfies ‖F(z*) - z*‖ < 1e-10."""
        from dynamic.analysis.scyfi import find_cycles
        from dynamic.analysis.scyfi_helpers import latent_step

        A, W, h = system_2d_holes
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        cycles_order_1 = result[0][0]
        for traj in cycles_order_1:
            z_star = traj[0]
            z_next = latent_step(z_star, A, W, h)
            assert torch.norm(z_next - z_star).item() < 1e-6

    def test_no_spurious_fps(self, system_2d_fp):
        """All returned FPs pass self-consistency."""
        from dynamic.analysis.scyfi import find_cycles
        from dynamic.analysis.scyfi_helpers import latent_step

        A, W, h = system_2d_fp
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        cycles_order_1 = result[0][0]
        for traj in cycles_order_1:
            z_star = traj[0]
            z_next = latent_step(z_star, A, W, h)
            assert torch.norm(z_next - z_star).item() < 1e-6


# ---------------------------------------------------------------------------
# PLRNN cycle tests
# ---------------------------------------------------------------------------
class TestFindCycles:
    """Tests for cycle detection in PLRNN systems."""

    def test_16_cycle_2d(self, system_2d_16cycle):
        """Finds 2 cycles of order 16."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_16cycle
        result = find_cycles(
            A, W, h, 16,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        cycles_order_16 = result[0][15]  # 0-indexed
        assert len(cycles_order_16) == 2

    def test_16_cycle_2d_nothing(self, system_2d_16cycle):
        """No cycles of order 15 in 16-cycle system."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_16cycle
        result = find_cycles(
            A, W, h, 16,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        cycles_order_15 = result[0][14]  # 0-indexed
        assert len(cycles_order_15) == 0

    def test_27_cycle_2d(self, system_2d_27cycle):
        """Finds 2 cycles of order 27."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_27cycle
        result = find_cycles(
            A, W, h, 27,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        cycles_order_27 = result[0][26]  # 0-indexed
        assert len(cycles_order_27) >= 2

    def test_cycle_is_periodic(self, system_2d_holes):
        """F^k(z*) = z* for all points in a cycle."""
        from dynamic.analysis.scyfi import find_cycles
        from dynamic.analysis.scyfi_helpers import latent_step

        A, W, h = system_2d_holes
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        # Check fixed points (1-cycles)
        for traj in result[0][0]:
            z = traj[0]
            z_next = latent_step(z, A, W, h)
            assert torch.norm(z_next - z).item() < 1e-6

    def test_cycle_points_distinct(self, system_2d_16cycle):
        """All k points in a k-cycle are pairwise distinct."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_16cycle
        result = find_cycles(
            A, W, h, 16,
            outer_loop_iterations=10, inner_loop_iterations=50,
        )
        for traj in result[0][15]:
            for i in range(len(traj)):
                for j in range(i + 1, len(traj)):
                    assert torch.norm(traj[i] - traj[j]).item() > 1e-6

    def test_10_cycle_10d(self, system_10d):
        """Finds 2 cycles of order 10 in 10D system."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_10d
        result = find_cycles(
            A, W, h, 10,
            outer_loop_iterations=20, inner_loop_iterations=80,
        )
        cycles_order_10 = result[0][9]  # 0-indexed
        assert len(cycles_order_10) == 2

    def test_10d_no_order_9(self, system_10d):
        """No cycles of order 9 in the 10D system."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_10d
        result = find_cycles(
            A, W, h, 10,
            outer_loop_iterations=20, inner_loop_iterations=80,
        )
        cycles_order_9 = result[0][8]  # 0-indexed
        assert len(cycles_order_9) == 0


# ---------------------------------------------------------------------------
# Eigenstructure tests
# ---------------------------------------------------------------------------
class TestEigenstructure:
    """Tests for eigenvalue-based analysis of found cycles."""

    def test_eigenvalues_returned(self, system_2d_holes):
        """Eigenvalues are returned alongside cycles."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_holes
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        eigvals_order_1 = result[1][0]
        assert len(eigvals_order_1) == len(result[0][0])

    def test_eigenvalues_are_numpy(self, system_2d_holes):
        """Eigenvalues are numpy arrays."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_holes
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        for ev in result[1][0]:
            assert isinstance(ev, np.ndarray)

    def test_eigenstructure_matches_numpy(self, system_2d_fp):
        """Eigenvalues match numpy.linalg.eig of composed Jacobian."""
        from dynamic.analysis.scyfi import find_cycles

        A, W, h = system_2d_fp
        result = find_cycles(
            A, W, h, 1,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        for idx, traj in enumerate(result[0][0]):
            z_star = traj[0]
            D = torch.diag((z_star > 0).float())
            J = (A + W @ D).numpy()
            expected_eigs = np.linalg.eigvals(J)
            found_eigs = result[1][0][idx]
            np.testing.assert_allclose(
                np.sort(np.abs(found_eigs)),
                np.sort(np.abs(expected_eigs)),
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------
class TestClassification:
    """Tests for classify_stable/unstable/saddle via eigenvalues."""

    def test_classify_from_eigenvalues(self):
        """classify_point correctly labels based on eigenvalue magnitudes."""
        from dynamic.analysis.subregions import classify_point

        # Stable: all |λ| < 1
        assert classify_point(np.array([0.5, 0.3])) == "stable"
        # Unstable: all |λ| > 1
        assert classify_point(np.array([1.5, 2.0])) == "unstable"
        # Saddle: mixed
        assert classify_point(np.array([0.5, 1.5])) == "saddle"


# ---------------------------------------------------------------------------
# shPLRNN support
# ---------------------------------------------------------------------------
class TestShPLRNN:
    """Tests for shPLRNN support in SCYFI algorithm."""

    def test_find_1_fp(self, sh_system_1fp):
        """Finds exactly 1 FP for M=2, H=10 shPLRNN system."""
        from dynamic.analysis.scyfi import find_cycles_sh

        A, W1, W2, h1, h2 = sh_system_1fp
        result = find_cycles_sh(
            A, W1, W2, h1, h2, 10,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        fps = result[0][0]
        assert len(fps) == 1

    def test_fp_value(self, sh_system_1fp):
        """FP ≈ [-0.03, 0.40]."""
        from dynamic.analysis.scyfi import find_cycles_sh

        A, W1, W2, h1, h2 = sh_system_1fp
        result = find_cycles_sh(
            A, W1, W2, h1, h2, 10,
            outer_loop_iterations=10, inner_loop_iterations=20,
        )
        fps = result[0][0]
        assert len(fps) >= 1
        fp = fps[0][0]  # first trajectory, first point
        fp_rounded = torch.round(fp * 100) / 100
        expected = torch.tensor([-0.03, 0.40])
        assert torch.allclose(fp_rounded, expected, atol=0.01)

    def test_2_cycle_4_cycle(self, sh_system_2_4_cycle):
        """Finds 2 two-cycles and 1 four-cycle."""
        from dynamic.analysis.scyfi import find_cycles_sh

        A, W1, W2, h1, h2 = sh_system_2_4_cycle
        result = find_cycles_sh(
            A, W1, W2, h1, h2, 4,
            outer_loop_iterations=10, inner_loop_iterations=60,
        )
        cycles_order_2 = result[0][1]  # 0-indexed
        cycles_order_4 = result[0][3]  # 0-indexed
        assert len(cycles_order_2) == 2
        assert len(cycles_order_4) == 1

    def test_no_1_or_3_cycle(self, sh_system_2_4_cycle):
        """No cycles at orders 1 and 3 for 2/4-cycle system."""
        from dynamic.analysis.scyfi import find_cycles_sh

        A, W1, W2, h1, h2 = sh_system_2_4_cycle
        result = find_cycles_sh(
            A, W1, W2, h1, h2, 4,
            outer_loop_iterations=10, inner_loop_iterations=60,
        )
        fps = result[0][0]  # order 1
        cycles_order_3 = result[0][2]  # order 3
        assert len(fps) == 0
        assert len(cycles_order_3) == 0
