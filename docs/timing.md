# Timing and computational cost

Historical 10,000-iteration timings compare an eight-core CPU exact-FEM loop
with one-GPU surrogate loops. Exact runs took about 1.7--2.2 hours. Surrogate
runs took about 1.8--4.0 minutes, corresponding to observed speed-up ranges of
26--34 (DeepONet), 34--44 (FNO), 49--63 (GenFNO) and 56--72 (GenDeepONet).
These are operational timings on different hardware and are not
hardware-normalised.

The 15,000 production fields required 15,000 forward and 240,000 tangent
solves. The recorded/extrapolated serial-equivalent data-generation time is
about 1.06 hours. Selected or architecture-matched training logs range from
2.5 to 27 hours on one A100, excluding model selection. Exact accounting is in
the Tier-A `revision/E9` files.
