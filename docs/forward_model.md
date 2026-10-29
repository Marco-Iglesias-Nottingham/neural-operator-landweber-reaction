# Forward problem and discretisation

On the unit square, the state solves

`-Delta p + m p = 1`, with homogeneous Dirichlet data on the boundary.

The inverse parameter is the physical reaction coefficient `m`, not its
logarithm. The inverse grid has 120 x 120 nodal values. Paper inversions use a
119 x 119 triangular P1 state mesh; synthetic truths were produced on a
952 x 952-cell fine mesh and transferred through the documented nested-grid
operators. Interior pressure vertices are the observations, with trapezoidal
quadrature weights.

`reaction_core/solver.py` implements the forward, tangent and adjoint solves;
`reaction_core/grid.py` and `reaction_core/inversion.py` implement transfers,
observation weights and parameter-space covariance actions.
