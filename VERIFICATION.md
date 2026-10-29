# Verification of the forward and derivative implementation

The finite-element implementation is checked in three independent ways:

1. **Mesh refinement:** `tests/test_reaction_core_and_grid.py` checks the
   forward solution and grid transfers on successively refined meshes.
2. **Finite-difference JVP:** `tests/test_reaction_core_and_grid.py` compares
   the tangent solve with a centred finite difference of the forward map.
3. **Adjoint identity:** the same test checks
   `(<DF(m)h,q> - <h,DF(m)^*q>)` relative to the paired norms. Production
   exact inversions also record an iteration-zero JVP/VJP identity check.

In a compatible DOLFINx environment, run:

```bash
python tests/test_reaction_core_and_grid.py --fem-nx 8 --fem-ny 8
python tests/test_identity_preconditioner.py
python tests/test_revision_matern_smoothness.py
```

Expected result: all tests report `OK`; the adjoint relative mismatch is below
`5e-8`. These are small verification problems and do not rerun the paper
experiments.

The exact solver assembles the coefficient-dependent state matrix once after
each coefficient update. The same PETSc matrix and KSP/preconditioner setup are
then reused for the forward, tangent and transpose-adjoint solves at that
iterate. The P1 parameter mass matrix/KSP and the observation-adjoint mass
matrix/KSP are assembled once for the whole run. Right-hand sides are freshly
assembled. There is no reuse of a coefficient-independent stiffness matrix as
a separate assembled object, no preconditioner-hierarchy reuse across changed
coefficients, and no warm start because output vectors are zeroed before each
solve.
