"""Thin Python wrapper for calling Julia SCYFI functions.

Handles numpy/torch ↔ Julia Array conversion and the axis-order
difference (Julia: dim×dim×order, Python: order×dim×dim).
"""

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


class JuliaScyfi:
    """Wrapper around the Julia SCYFI module.

    Parameters
    ----------
    jl : juliacall.Main
        An already-initialised Julia session with SCYFI loaded.
    """

    def __init__(self, jl):
        self.jl = jl

    # ------------------------------------------------------------------
    # Type conversion helpers
    # ------------------------------------------------------------------
    def _to_julia_matrix(self, t: Tensor) -> object:
        """Convert (M, M) torch tensor → Julia Matrix{Float64}."""
        arr = t.detach().double().numpy()
        return self.jl.seval("Matrix{Float64}")(arr)

    def _to_julia_vector(self, t: Tensor) -> object:
        """Convert (M,) torch tensor → Julia Vector{Float64}."""
        arr = t.detach().double().numpy()
        return self.jl.seval("Vector{Float64}")(arr)

    def _to_julia_d_list(self, d_list: Tensor) -> object:
        """Convert (order, dim, dim) → Julia (dim, dim, order) 3D array."""
        arr = d_list.detach().double().numpy().transpose(1, 2, 0).copy()
        return self.jl.seval("Array{Float64,3}")(arr)

    def _from_julia_matrix(self, jl_mat) -> Tensor:
        """Julia Matrix/Diagonal/UniformScaling → (M, M) float64 tensor."""
        # Materialise abstract types
        mat = self.jl.seval("x -> Matrix{Float64}(x)")(jl_mat)
        arr = np.array(mat, dtype=np.float64)
        return torch.from_numpy(arr)

    def _from_julia_vector(self, jl_vec) -> Tensor:
        """Julia Vector → (M,) float64 tensor."""
        arr = np.array(jl_vec, dtype=np.float64)
        return torch.from_numpy(arr)

    def _from_julia_eigvals(self, jl_eigs) -> ndarray:
        """Julia eigenvalue array → numpy array."""
        return np.array(jl_eigs, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Wrapped helper functions
    # ------------------------------------------------------------------
    def construct_relu_matrix(self, quadrant: int, dim: int) -> Tensor:
        """Call Julia construct_relu_matrix."""
        result = self.jl.seval(
            "(q, d) -> Array(SCYFI.construct_relu_matrix(Int128(q), d))"
        )(quadrant, dim)
        return self._from_julia_matrix(result)

    def get_factor_in_front_of_z(
        self, A: Tensor, W: Tensor, D_list: Tensor, order: int
    ) -> Tensor:
        """Call Julia get_factor_in_front_of_z."""
        result = self.jl.seval(
            "(A,W,D,o) -> Matrix{Float64}(SCYFI.get_factor_in_front_of_z(A,W,D,o))"
        )(
            self._to_julia_matrix(A),
            self._to_julia_matrix(W),
            self._to_julia_d_list(D_list),
            order,
        )
        arr = np.array(result, dtype=np.float64)
        return torch.from_numpy(arr)

    def get_factor_in_front_of_h(
        self, A: Tensor, W: Tensor, D_list: Tensor, order: int
    ) -> Tensor:
        """Call Julia get_factor_in_front_of_h.

        Note: Julia convention is to receive D_list[:,:,2:end] (pre-sliced),
        while Python's version takes the full D_list and skips index 0.
        We slice D_list[1:] before passing to Julia.
        """
        dim = A.shape[0]
        # Slice D_list to match Julia's convention: D_list[:,:,2:end]
        d_sliced = D_list[1:]  # Python (order, dim, dim) → skip first
        result = self.jl.seval(
            """(A,W,D,o,dim) -> begin
                r = SCYFI.get_factor_in_front_of_h(A,W,D,o)
                if r isa LinearAlgebra.UniformScaling
                    Matrix{Float64}(r, dim, dim)
                else
                    Matrix{Float64}(r)
                end
            end"""
        )(
            self._to_julia_matrix(A),
            self._to_julia_matrix(W),
            self._to_julia_d_list(d_sliced),
            order,
            dim,
        )
        arr = np.array(result, dtype=np.float64)
        return torch.from_numpy(arr)

    def get_cycle_point_candidate(
        self, A: Tensor, W: Tensor, D_list: Tensor, h: Tensor, order: int
    ) -> Tensor | None:
        """Call Julia get_cycle_point_candidate."""
        result = self.jl.seval("SCYFI.get_cycle_point_candidate")(
            self._to_julia_matrix(A),
            self._to_julia_matrix(W),
            self._to_julia_d_list(D_list),
            self._to_julia_vector(h),
            order,
        )
        if result is None or self.jl.seval("isnothing")(result):
            return None
        return self._from_julia_vector(result)

    def get_eigvals(
        self, A: Tensor, W: Tensor, D_list: Tensor, order: int
    ) -> ndarray:
        """Call Julia get_eigvals."""
        result = self.jl.seval("SCYFI.get_eigvals")(
            self._to_julia_matrix(A),
            self._to_julia_matrix(W),
            self._to_julia_d_list(D_list),
            order,
        )
        return self._from_julia_eigvals(result)

    def latent_step(
        self, z: Tensor, A: Tensor, W: Tensor, h: Tensor
    ) -> Tensor:
        """Call Julia latent_step."""
        result = self.jl.seval("SCYFI.latent_step")(
            self._to_julia_vector(z),
            self._to_julia_matrix(A),
            self._to_julia_matrix(W),
            self._to_julia_vector(h),
        )
        return self._from_julia_vector(result)

    def get_latent_time_series(
        self,
        time_steps: int,
        A: Tensor,
        W: Tensor,
        h: Tensor,
        dim: int,
        *,
        z_0: Tensor | None = None,
    ) -> list[Tensor]:
        """Call Julia get_latent_time_series."""
        if z_0 is not None:
            result = self.jl.seval(
                """(ts, A, W, h, dim, z0) ->
                    SCYFI.get_latent_time_series(ts, A, W, h, dim, z_0=z0)
                """
            )(
                time_steps,
                self._to_julia_matrix(A),
                self._to_julia_matrix(W),
                self._to_julia_vector(h),
                dim,
                self._to_julia_vector(z_0),
            )
        else:
            result = self.jl.seval("SCYFI.get_latent_time_series")(
                time_steps,
                self._to_julia_matrix(A),
                self._to_julia_matrix(W),
                self._to_julia_vector(h),
                dim,
            )
        return [self._from_julia_vector(result[i]) for i in range(len(result))]

    # ------------------------------------------------------------------
    # shPLRNN helpers
    # ------------------------------------------------------------------
    def get_factors_sh(
        self, A: Tensor, W1: Tensor, W2: Tensor, D_list: Tensor, order: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call Julia get_factors_sh."""
        result = self.jl.seval(
            """(A, W1, W2, D_list, order) ->
                SCYFI.get_factors(
                    Diagonal(A), W1, W2, D_list, order
                )
            """
        )(
            self._to_julia_vector(A),
            self._to_julia_matrix(W1),
            self._to_julia_matrix(W2),
            self._to_julia_d_list(D_list),
            order,
        )
        return (
            self._from_julia_matrix(result[0]),
            self._from_julia_matrix(result[1]),
            self._from_julia_matrix(result[2]),
        )

    def get_eigvals_sh(
        self, A: Tensor, W1: Tensor, W2: Tensor, D_list: Tensor, order: int,
    ) -> ndarray:
        """Call Julia get_eigvals for shPLRNN."""
        result = self.jl.seval(
            """(A, W1, W2, D_list, order) ->
                SCYFI.get_eigvals(
                    Diagonal(A), W1, W2, D_list, order
                )
            """
        )(
            self._to_julia_vector(A),
            self._to_julia_matrix(W1),
            self._to_julia_matrix(W2),
            self._to_julia_d_list(D_list),
            order,
        )
        return self._from_julia_eigvals(result)

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------
    def find_cycles(
        self,
        A: Tensor,
        W: Tensor,
        h: Tensor,
        max_order: int,
        *,
        outer_loop_iterations: int = 10,
        inner_loop_iterations: int = 20,
    ) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
        """Call Julia find_cycles (top-level)."""
        result = self.jl.seval(
            """(A, W, h, max_order, outer, inner) -> begin
                all_cycles = Array[]
                all_eigvals = Array[]
                for order in 1:max_order
                    cycles, evs = SCYFI.scy_fi(
                        A, W, h, order, all_cycles,
                        outer_loop_iterations=outer,
                        inner_loop_iterations=inner,
                    )
                    push!(all_cycles, cycles)
                    push!(all_eigvals, evs)
                end
                return (all_cycles, all_eigvals)
            end"""
        )(
            self._to_julia_matrix(A),
            self._to_julia_matrix(W),
            self._to_julia_vector(h),
            max_order,
            outer_loop_iterations,
            inner_loop_iterations,
        )
        return self._parse_cycles_result(result)

    def find_cycles_sh(
        self,
        A: Tensor,
        W1: Tensor,
        W2: Tensor,
        h1: Tensor,
        h2: Tensor,
        max_order: int,
        *,
        outer_loop_iterations: int = 10,
        inner_loop_iterations: int = 20,
    ) -> tuple[list[list[list[Tensor]]], list[list[ndarray]]]:
        """Call Julia find_cycles for shPLRNN."""
        result = self.jl.seval(
            """(A, W1, W2, h1, h2, max_order, outer, inner) -> begin
                all_cycles = Array[]
                all_eigvals = Array[]
                for order in 1:max_order
                    cycles, evs = SCYFI.scy_fi(
                        Diagonal(A), W1, W2, h1, h2, order, all_cycles,
                        outer_loop_iterations=outer,
                        inner_loop_iterations=inner,
                    )
                    push!(all_cycles, cycles)
                    push!(all_eigvals, evs)
                end
                return (all_cycles, all_eigvals)
            end"""
        )(
            self._to_julia_vector(A),
            self._to_julia_matrix(W1),
            self._to_julia_matrix(W2),
            self._to_julia_vector(h1),
            self._to_julia_vector(h2),
            max_order,
            outer_loop_iterations,
            inner_loop_iterations,
        )
        return self._parse_cycles_result(result)

    def _parse_cycles_result(self, result):
        """Parse Julia (all_cycles, all_eigvals) into Python structures."""
        jl_cycles, jl_eigvals = result[0], result[1]
        all_cycles = []
        all_eigvals = []

        for order_idx in range(len(jl_cycles)):
            order_cycles = []
            order_eigvals = []
            cycles_at_order = jl_cycles[order_idx]
            eigvals_at_order = jl_eigvals[order_idx]

            for c_idx in range(len(cycles_at_order)):
                traj = cycles_at_order[c_idx]
                trajectory = [
                    self._from_julia_vector(traj[i])
                    for i in range(len(traj))
                ]
                order_cycles.append(trajectory)
                order_eigvals.append(
                    self._from_julia_eigvals(eigvals_at_order[c_idx])
                )

            all_cycles.append(order_cycles)
            all_eigvals.append(order_eigvals)

        return all_cycles, all_eigvals
