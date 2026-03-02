"""Cross-validation conftest — Julia session management and fixtures."""

import os
import shutil
import sys

import numpy as np
import pytest
import torch

# Mark all tests in this directory as requiring Julia
pytestmark = pytest.mark.julia

# ---------------------------------------------------------------------------
# Julia availability check
# ---------------------------------------------------------------------------
JULIA_AVAILABLE = shutil.which("julia") is not None or os.path.exists(
    os.path.expanduser("~/.juliaup/bin/julia")
)


def _get_julia():
    """Lazily initialise the Julia session and load SCYFI."""
    # Ensure juliaup Julia is on PATH
    juliaup_bin = os.path.expanduser("~/.juliaup/bin")
    if juliaup_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = juliaup_bin + ":" + os.environ.get("PATH", "")

    from juliacall import Main as jl

    scyfi_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "reference", "SCYFI")
    )

    # Activate and load the SCYFI project
    jl.seval(f"""
        import Pkg
        Pkg.activate("{scyfi_path}")
        include("{scyfi_path}/src/SCYFI.jl")
        using .SCYFI
        using LinearAlgebra
    """)
    return jl


@pytest.fixture(scope="session")
def jl_session():
    """Session-scoped Julia fixture. Skips all tests if Julia unavailable."""
    if not JULIA_AVAILABLE:
        pytest.skip("Julia not installed — skipping cross-validation tests")
    try:
        jl = _get_julia()
    except Exception as e:
        pytest.skip(f"Julia setup failed: {e}")
    return jl


# ---------------------------------------------------------------------------
# Shared parameter fixtures (identical to test_scyfi.py)
# ---------------------------------------------------------------------------
@pytest.fixture()
def system_2d_holes():
    """2D PLRNN with 1 FP — 'holes' test case."""
    A = torch.tensor([[0.7, 0.0], [0.0, -0.7]], dtype=torch.float64)
    W = torch.tensor([[0.0, -0.14375], [0.52505308, 0.0]], dtype=torch.float64)
    h = torch.tensor([0.37298253, -0.97931491], dtype=torch.float64)
    return A, W, h


@pytest.fixture()
def system_2d_fp():
    """2D PLRNN with 1 FP and known value."""
    A = torch.tensor(
        [[-0.022231099999999948, 0.0], [0.0, 0.96461297]], dtype=torch.float64
    )
    W = torch.tensor(
        [[0.0, -0.6437499999999996], [0.52505308, 0.0]], dtype=torch.float64
    )
    h = torch.tensor([0.37298253000000087, -0.97931491], dtype=torch.float64)
    return A, W, h


@pytest.fixture()
def system_2d_16cycle():
    """2D PLRNN with 16-cycles."""
    A = torch.tensor(
        [[0.8377689000000008, 0.0], [0.0, 0.96461297]], dtype=torch.float64
    )
    W = torch.tensor(
        [[0.0, -0.6437499999999996], [0.52505308, 0.0]], dtype=torch.float64
    )
    h = torch.tensor([0.37298253000000087, -0.97931491], dtype=torch.float64)
    return A, W, h


@pytest.fixture()
def sh_system_1fp():
    """shPLRNN M=2, H=10 system with 1 FP."""
    A = torch.tensor(
        [1.4799836250879086, -1.6945099980546405], dtype=torch.float64
    )
    W1 = torch.tensor(
        [
            [
                0.8900829721936014, 0.24766868576521212, 0.2554263710950393,
                -0.5771080679179108, -2.129379740761085, 0.031728042646673976,
                -0.7740703525170001, 0.04768259094291391, 0.6792959504153095,
                -1.7781119769858111,
            ],
            [
                0.6297279024943379, 0.5586975133684497, -0.18842627535543358,
                0.7152028558426154, 2.238266026927219, 1.0163441695005024,
                0.6115773461504028, 1.8513040760318145, 0.5127750227966122,
                -2.337416929726536,
            ],
        ],
        dtype=torch.float64,
    )
    W2 = torch.tensor(
        [
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
        ],
        dtype=torch.float64,
    )
    h1 = torch.tensor(
        [0.18005923664630755, 0.42057092737475477], dtype=torch.float64
    )
    h2 = torch.tensor(
        [
            -0.017897789466083707, 1.014731429962641, -0.3431368102401408,
            -2.3065949842414866, -0.7738820872835147, 0.009176181040445722,
            0.6395596355828497, -0.3762485593614053, -0.12524612444111408,
            0.25769322230050085,
        ],
        dtype=torch.float64,
    )
    return A, W1, W2, h1, h2
