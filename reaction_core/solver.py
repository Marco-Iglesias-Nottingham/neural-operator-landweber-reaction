from __future__ import annotations

from typing import Any, Optional

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem import petsc

from .config import ReactionConfig


# The reaction problem is scalar, so a state/gradient is already represented
# by one DOLFINx Function.  These aliases make the public API self-documenting
# without wrapping Functions and obscuring their PETSc vectors.
ReactionState = fem.Function
ReactionGradient = fem.Function


class ReactionSolver2D:
    r"""Cached finite-element solver for the reaction-coefficient problem.

    The authoritative model is

    .. math::

        -\Delta p + u p = f \quad\hbox{in } (0,L_x)\times(0,L_y),
        \qquad p=0 \quad\hbox{on } \partial\Omega.

    ``u`` is a continuous P1 finite-element function and must remain above the
    solver's configured nonnegative nodal lower bound. ``ReactionConfig`` uses
    the historical default 10, while the paper training and inversion
    pipelines explicitly configure the admissible lower bound as 0. The state
    may use P1 or P2 Lagrange elements.

    For a physical coefficient direction ``h``, the tangent and adjoint are

    .. math::

        A(u)\,\dot p = -h p,
        \qquad A(u)^T\lambda = q.

    Here the adjoint right-hand side represents the L2 pairing with ``q``.
    The corresponding physical L2 gradient is the P1 mass projection of
    ``-p*lambda``.

    The state matrix and its direct-solver setup are cached.  Changing ``u``
    invalidates the cache; forward, tangent and adjoint solves at the same
    coefficient reuse the same assembled matrix and LU factorisation.
    """

    HARD_MINIMUM_U = 0.0

    def __init__(
        self,
        config: ReactionConfig | None = None,
        *,
        comm=MPI.COMM_WORLD,
        minimum_u: float | None = None,
        initial_u: float = 50.0,
        solver_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config or ReactionConfig()
        self.config.validate()
        self.comm = comm

        configured_minimum = getattr(self.config, "u_min", self.HARD_MINIMUM_U)
        requested_minimum = (
            configured_minimum if minimum_u is None else float(minimum_u)
        )
        self.minimum_u = max(self.HARD_MINIMUM_U, float(requested_minimum))
        if not np.isfinite(self.minimum_u):
            raise ValueError("minimum_u must be finite.")
        if not np.isfinite(initial_u) or float(initial_u) < self.minimum_u:
            raise ValueError(
                f"initial_u must be finite and at least {self.minimum_u:g}."
            )

        cfg = self.config
        self.domain = mesh.create_rectangle(
            comm,
            [np.array([0.0, 0.0]), np.array([cfg.Lx, cfg.Ly])],
            [int(cfg.fem_nx), int(cfg.fem_ny)],
            cell_type=mesh.CellType.triangle,
        )
        self.V = fem.functionspace(
            self.domain,
            ("Lagrange", int(cfg.state_degree)),
        )
        # Keep the physical reaction coefficient nodal, matching the legacy
        # inverse problem and the regular neural grid exactly.
        self.Q = fem.functionspace(self.domain, ("Lagrange", 1))

        self.u = fem.Function(self.Q)
        self.u.name = "reaction_coefficient_u"
        self.u.x.array[:] = float(initial_u)
        self.u.x.scatter_forward()
        # Compatibility with the legacy root-level solver.
        self.u_fun = self.u
        self.coefficient = self.u

        self.source = fem.Constant(
            self.domain,
            PETSc.ScalarType(float(cfg.source_value)),
        )

        self._build_boundary_condition()
        self._build_forms()

        self.solver_options = dict(
            solver_options
            or {
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        )
        self._prefix_counter = 0

        self._operator_dirty = True
        self._operator_assembly_count = 0
        self._state_linear_solve_count = 0
        self._forward_solve_count = 0
        self._tangent_solve_count = 0
        self._adjoint_solve_count = 0
        self._projection_solve_count = 0
        self._coordinate_adjoint_assembly_count = 0

        self._state_matrix: PETSc.Mat | None = None
        self._state_ksp: PETSc.KSP | None = None
        self._Q_mass_matrix: PETSc.Mat | None = None
        self._Q_mass_ksp: PETSc.KSP | None = None

        self._build_cached_state_solver()
        self._build_cached_parameter_projection()

    # ------------------------------------------------------------------
    # Spaces, boundary conditions and reusable forms
    # ------------------------------------------------------------------

    def _build_boundary_condition(self) -> None:
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            self.domain,
            fdim,
            lambda x: np.ones(x.shape[1], dtype=bool),
        )
        boundary_dofs = fem.locate_dofs_topological(
            self.V,
            fdim,
            boundary_facets,
        )
        zero = fem.Function(self.V)
        zero.name = "zero_boundary_value"
        zero.x.array[:] = 0.0
        zero.x.scatter_forward()
        self.bc = fem.dirichletbc(zero, boundary_dofs)
        self.bcs = [self.bc]
        self.boundary_dofs = np.asarray(boundary_dofs, dtype=np.int32)

    def _build_forms(self) -> None:
        trial = ufl.TrialFunction(self.V)
        test = ufl.TestFunction(self.V)

        self._a_form = fem.form(
            (
                ufl.inner(ufl.grad(trial), ufl.grad(test))
                + self.u * trial * test
            )
            * ufl.dx
        )
        self._forward_L_form = fem.form(self.source * test * ufl.dx)

        # Reusable tangent buffers and form.
        self._tangent_state = fem.Function(self.V)
        self._tangent_direction = fem.Function(self.Q)
        self._tangent_L_form = fem.form(
            -self._tangent_direction * self._tangent_state * test * ufl.dx
        )

        # Reusable L2-adjoint covector and form.
        self._adjoint_covector = fem.Function(self.V)
        self._adjoint_L_form = fem.form(
            self._adjoint_covector * test * ufl.dx
        )

        # Reusable physical-gradient projection form.  Since Q is P1 rather
        # than DG0, its mass matrix is not diagonal and must be solved exactly.
        self._gradient_state = fem.Function(self.V)
        self._gradient_adjoint = fem.Function(self.V)
        q_test = ufl.TestFunction(self.Q)
        self._gradient_L_form = fem.form(
            -self._gradient_state * self._gradient_adjoint * q_test * ufl.dx
        )

    # ------------------------------------------------------------------
    # Coefficient and Function utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_function_values(
        destination: fem.Function,
        source: fem.Function,
    ) -> None:
        if destination.x.array.shape != source.x.array.shape:
            raise ValueError(
                "Function storage shapes differ: "
                f"{destination.x.array.shape} and {source.x.array.shape}."
            )
        destination.x.array[:] = source.x.array
        destination.x.scatter_forward()

    @staticmethod
    def _set_function_values(
        destination: fem.Function,
        values,
        *,
        name: str,
    ) -> None:
        if isinstance(values, fem.Function):
            ReactionSolver2D._copy_function_values(destination, values)
            return
        array = np.asarray(values, dtype=destination.x.array.dtype)
        if array.shape != destination.x.array.shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected "
                f"{destination.x.array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")
        destination.x.array[:] = array
        destination.x.scatter_forward()

    def coefficient_coordinates(self) -> np.ndarray:
        return self.Q.tabulate_dof_coordinates()

    # MRE-compatible name used by generic grid helpers.
    parameter_coordinates = coefficient_coordinates

    def state_coordinates(self) -> np.ndarray:
        return self.V.tabulate_dof_coordinates()

    def new_parameter_function(
        self,
        values=None,
        *,
        name: str = "coefficient_direction",
    ) -> fem.Function:
        output = fem.Function(self.Q)
        output.name = name
        if values is not None:
            self._set_function_values(output, values, name=name)
        return output

    new_coefficient_function = new_parameter_function

    def new_state_function(
        self,
        values=None,
        *,
        name: str = "reaction_state",
    ) -> fem.Function:
        output = fem.Function(self.V)
        output.name = name
        if values is not None:
            self._set_function_values(output, values, name=name)
        return output

    def set_coefficient(self, values) -> None:
        if isinstance(values, fem.Function):
            array = np.asarray(values.x.array, dtype=self.u.x.array.dtype)
        else:
            array = np.asarray(values, dtype=self.u.x.array.dtype)
        if array.shape != self.u.x.array.shape:
            raise ValueError(
                f"u has shape {array.shape}; expected {self.u.x.array.shape}."
            )
        finite_everywhere = bool(
            self.comm.allreduce(
                bool(np.all(np.isfinite(array))),
                op=MPI.LAND,
            )
        )
        if not finite_everywhere:
            raise ValueError("u contains non-finite values.")
        minimum = float(
            self.comm.allreduce(float(np.min(array)), op=MPI.MIN)
        )
        if minimum < self.minimum_u:
            raise ValueError(
                "The reaction coefficient violates its non-negativity safeguard: "
                f"min(u)={minimum:.8g} < {self.minimum_u:.8g}."
            )

        changed = not np.array_equal(array, self.u.x.array)
        self.u.x.array[:] = array
        self.u.x.scatter_forward()
        if changed:
            self._operator_dirty = True

    set_u = set_coefficient
    set_u_array = set_coefficient
    set_coefficient_array = set_coefficient

    def get_coefficient_array(self) -> np.ndarray:
        return self.u.x.array.copy()

    def set_constant_coefficient(self, value: float) -> None:
        value = float(value)
        self.set_coefficient(np.full_like(self.u.x.array, value))

    # ------------------------------------------------------------------
    # Cached matrix and state solves
    # ------------------------------------------------------------------

    def _prefix(self, label: str) -> str:
        self._prefix_counter += 1
        return f"reaction_{label}_{self._prefix_counter}_"

    @staticmethod
    def _apply_petsc_options(ksp: PETSc.KSP, options: dict[str, Any]) -> None:
        prefix = ksp.getOptionsPrefix() or ""
        database = PETSc.Options()
        database.prefixPush(prefix)
        try:
            for key, value in options.items():
                database[key] = value
            ksp.setFromOptions()
        finally:
            for key in options:
                try:
                    del database[key]
                except KeyError:
                    pass
            database.prefixPop()

    def _build_cached_state_solver(self) -> None:
        self._state_matrix = petsc.create_matrix(self._a_form)
        self._state_ksp = PETSc.KSP().create(self.comm)
        self._state_ksp.setOptionsPrefix(self._prefix("cached_state"))
        self._state_ksp.setOperators(self._state_matrix)
        self._apply_petsc_options(self._state_ksp, self.solver_options)
        self._operator_dirty = True

    def _ensure_state_operator(self) -> None:
        if not self._operator_dirty:
            return
        assert self._state_matrix is not None
        assert self._state_ksp is not None

        self._state_matrix.zeroEntries()
        petsc.assemble_matrix(
            self._state_matrix,
            self._a_form,
            bcs=self.bcs,
        )
        self._state_matrix.assemble()
        self._state_ksp.setOperators(self._state_matrix)
        # For a direct solver this performs the numeric factorisation now;
        # subsequent solves only replace the right-hand side.
        self._state_ksp.setUp()

        self._operator_dirty = False
        self._operator_assembly_count += 1

    def _assemble_state_rhs(self, L_form) -> PETSc.Vec:
        vector = petsc.assemble_vector(L_form)
        petsc.apply_lifting(
            vector,
            [self._a_form],
            bcs=[self.bcs],
        )
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        petsc.set_bc(vector, self.bcs)
        return vector

    def _solve_state_system(
        self,
        L_form,
        *,
        transpose: bool,
        name: str,
        solve_kind: str,
        out: fem.Function | None = None,
    ) -> fem.Function:
        self._ensure_state_operator()
        assert self._state_ksp is not None

        rhs = self._assemble_state_rhs(L_form)
        output = out if out is not None else fem.Function(self.V)
        output.name = name
        output.x.array[:] = 0.0
        output.x.scatter_forward()

        try:
            if transpose:
                self._state_ksp.solveTranspose(rhs, output.x.petsc_vec)
            else:
                self._state_ksp.solve(rhs, output.x.petsc_vec)
            reason = self._state_ksp.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(
                    f"Cached reaction {solve_kind} solve failed: "
                    f"KSP converged reason={reason}."
                )
            output.x.scatter_forward()
        finally:
            rhs.destroy()

        self._state_linear_solve_count += 1
        if solve_kind == "forward":
            self._forward_solve_count += 1
        elif solve_kind == "tangent":
            self._tangent_solve_count += 1
        elif solve_kind == "adjoint":
            self._adjoint_solve_count += 1
        return output

    def forward(self, coefficient=None, out: fem.Function | None = None) -> fem.Function:
        if coefficient is not None:
            self.set_coefficient(coefficient)
        return self._solve_state_system(
            self._forward_L_form,
            transpose=False,
            name="pressure_p",
            solve_kind="forward",
            out=out,
        )

    solve_forward = forward
    solve = forward

    def tangent(
        self,
        state: fem.Function,
        direction,
        out: fem.Function | None = None,
    ) -> fem.Function:
        self._copy_function_values(self._tangent_state, state)
        self._set_function_values(
            self._tangent_direction,
            direction,
            name="physical coefficient direction",
        )
        return self._solve_state_system(
            self._tangent_L_form,
            transpose=False,
            name="pressure_tangent_dp",
            solve_kind="tangent",
            out=out,
        )

    solve_linearized = tangent

    def adjoint(
        self,
        covector,
        out: fem.Function | None = None,
    ) -> fem.Function:
        self._set_function_values(
            self._adjoint_covector,
            covector,
            name="state L2 covector",
        )
        return self._solve_state_system(
            self._adjoint_L_form,
            transpose=True,
            name="pressure_adjoint_lambda",
            solve_kind="adjoint",
            out=out,
        )

    solve_adjoint = adjoint

    # ------------------------------------------------------------------
    # Exact P1 L2 projection of the physical gradient -p*lambda
    # ------------------------------------------------------------------

    def _build_cached_parameter_projection(self) -> None:
        trial = ufl.TrialFunction(self.Q)
        test = ufl.TestFunction(self.Q)
        self._Q_mass_form = fem.form(trial * test * ufl.dx)
        self._Q_mass_matrix = petsc.assemble_matrix(self._Q_mass_form)
        self._Q_mass_matrix.assemble()

        self._Q_mass_ksp = PETSc.KSP().create(self.comm)
        self._Q_mass_ksp.setOptionsPrefix(
            self._prefix("cached_parameter_mass")
        )
        self._Q_mass_ksp.setOperators(self._Q_mass_matrix)
        self._Q_mass_ksp.setType("preonly")
        self._Q_mass_ksp.getPC().setType("lu")
        self._Q_mass_ksp.setFromOptions()
        self._Q_mass_ksp.setUp()

    def parameter_gradient(
        self,
        state: fem.Function,
        adjoint: fem.Function,
        *,
        out: fem.Function | None = None,
    ) -> fem.Function:
        self._copy_function_values(self._gradient_state, state)
        self._copy_function_values(self._gradient_adjoint, adjoint)
        assert self._Q_mass_ksp is not None

        rhs = petsc.assemble_vector(self._gradient_L_form)
        output = out if out is not None else fem.Function(self.Q)
        output.name = "physical_L2_gradient_u"
        output.x.array[:] = 0.0
        output.x.scatter_forward()
        try:
            rhs.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            self._Q_mass_ksp.solve(rhs, output.x.petsc_vec)
            reason = self._Q_mass_ksp.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(
                    "P1 parameter mass projection failed: "
                    f"KSP converged reason={reason}."
                )
            output.x.scatter_forward()
        finally:
            rhs.destroy()
        self._projection_solve_count += 1
        return output

    def parameter_coordinate_adjoint(
        self,
        state: fem.Function,
        adjoint: fem.Function,
    ) -> np.ndarray:
        r"""Return the exact derivative with respect to nodal physical ``u``.

        If ``u = sum_i u_i phi_i`` and ``lambda`` is the state adjoint, this
        returns the coefficient-space load vector

        .. math::

            c_i = -\int_\Omega p\,\lambda\,\phi_i\,dx.

        Consequently ``c @ h`` is the derivative in a vector of P1 nodal
        coordinates ``h``.  Unlike :meth:`parameter_gradient`, this method
        deliberately does *not* apply the inverse parameter mass matrix.  It
        is therefore the quantity to pass to a nodal covariance matrix in a
        covariance-preconditioned Landweber step.

        The returned array follows the native ``Q`` DOF order.  The
        structured coefficient map can permute it to the regular ``(nx,ny)``
        grid.  This coordinate-returning convenience is intentionally serial;
        distributed callers need an explicit global ownership/gather policy.
        """
        if self.comm.size != 1:
            raise RuntimeError(
                "parameter_coordinate_adjoint currently requires one MPI "
                "rank; a distributed coordinate vector needs an explicit "
                "global ownership policy."
            )

        self._copy_function_values(self._gradient_state, state)
        self._copy_function_values(self._gradient_adjoint, adjoint)
        load = petsc.assemble_vector(self._gradient_L_form)
        try:
            load.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            values = np.asarray(load.getArray(readonly=True)).copy()
        finally:
            load.destroy()

        expected_shape = self.u.x.array.shape
        if values.shape != expected_shape:
            raise RuntimeError(
                "The serial parameter load has shape "
                f"{values.shape}, but the coefficient storage has shape "
                f"{expected_shape}."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                "The assembled parameter coordinate adjoint is non-finite."
            )
        self._coordinate_adjoint_assembly_count += 1
        return values

    physical_parameter_coordinate_adjoint = parameter_coordinate_adjoint

    def apply_jacobian_adjoint(
        self,
        state: fem.Function,
        covector,
    ) -> tuple[fem.Function, fem.Function]:
        adjoint = self.adjoint(covector)
        gradient = self.parameter_gradient(state, adjoint)
        return gradient, adjoint

    physical_gradient = parameter_gradient

    # ------------------------------------------------------------------
    # Inner products, diagnostics and lifecycle
    # ------------------------------------------------------------------

    def _assemble_global(self, expression) -> float:
        local = fem.assemble_scalar(fem.form(expression))
        return float(self.comm.allreduce(local, op=MPI.SUM))

    def state_inner(self, left: fem.Function, right: fem.Function) -> float:
        return self._assemble_global(left * right * ufl.dx)

    def state_norm(self, state: fem.Function) -> float:
        return float(np.sqrt(max(self.state_inner(state, state), 0.0)))

    def parameter_inner(self, left: fem.Function, right: fem.Function) -> float:
        return self._assemble_global(left * right * ufl.dx)

    def parameter_norm(self, parameter: fem.Function) -> float:
        return float(
            np.sqrt(max(self.parameter_inner(parameter, parameter), 0.0))
        )

    def cache_diagnostics(self) -> dict[str, int | bool]:
        return {
            "operator_dirty": bool(self._operator_dirty),
            "operator_assemblies_factorizations": int(
                self._operator_assembly_count
            ),
            "state_linear_solves": int(self._state_linear_solve_count),
            "forward_solves": int(self._forward_solve_count),
            "tangent_solves": int(self._tangent_solve_count),
            "adjoint_solves": int(self._adjoint_solve_count),
            "parameter_projection_solves": int(self._projection_solve_count),
            "parameter_coordinate_adjoint_assemblies": int(
                self._coordinate_adjoint_assembly_count
            ),
        }

    def num_state_dofs(self) -> int:
        return int(self.V.dofmap.index_map.size_global)

    def num_parameter_dofs(self) -> int:
        return int(self.Q.dofmap.index_map.size_global)

    # Legacy convenience name referred to state DOFs.
    num_dofs = num_state_dofs

    def close(self) -> None:
        if self._state_ksp is not None:
            self._state_ksp.destroy()
            self._state_ksp = None
        if self._state_matrix is not None:
            self._state_matrix.destroy()
            self._state_matrix = None
        if self._Q_mass_ksp is not None:
            self._Q_mass_ksp.destroy()
            self._Q_mass_ksp = None
        if self._Q_mass_matrix is not None:
            self._Q_mass_matrix.destroy()
            self._Q_mass_matrix = None

    destroy = close

    def __enter__(self) -> "ReactionSolver2D":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may already have torn PETSc down.
            pass


# Short compatibility name for scripts ported from the legacy root project.
ForwardSolver = ReactionSolver2D


__all__ = [
    "ForwardSolver",
    "ReactionGradient",
    "ReactionSolver2D",
    "ReactionState",
]
