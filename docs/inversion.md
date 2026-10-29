# Landweber inversion

The exact and surrogate drivers are `scripts/inversion/landweber_exact.py` and
`scripts/inversion/landweber_surrogate.py`. Both use the same 120 x 120
physical parameter grid, initial field `m_0=100`, box `[0,200]`, adaptive step
rule, physical-change cap and 10,000-iteration paper budget.

The baseline parameter-space geometry is the nodal action of a non-periodic
Matérn covariance with `nu_C=3/2` and `ell_C=0.1`. It is applied without an
extra grid-cell quadrature factor because the parameter vector and adjoint are
coordinate-space nodal quantities. Identity and low-smoothness geometries are
controlled ablations. Noisy experiments save full trajectories; discrepancy
crossings are evaluated retrospectively rather than used for early stopping.
