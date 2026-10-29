# Reproducibility

## Tier A: paper products

Tier A consumes the checksummed `reaction-paper-v1` bundle and regenerates all
seven Reaction tables and five figures. Tables must be byte-identical to the
frozen references. PDFs are rendered at 200 dpi and compared pixel by pixel;
the strict audit tolerance is zero differing pixels. The plotted arrays and
all bundle files have SHA-256 checksums.

Expected wall time is a few minutes on a laptop, dominated by rendering the
three convergence figures. Run `python reproduce/paper.py`.

The release-preparation run verified all 22 Tier-A data files, reproduced all
seven tables byte-for-byte, matched all 44 computation-derived prose claims,
and reproduced Figures 1, 3 and 5 with zero differing pixels. Figures 2 and 4
use the exact frozen arrays and numerical labels, but the portable environment
did not recreate their text/layout pixel-for-pixel: 8,252 and 138,045 rendered
pixels differed at 200 dpi. The authors accepted these two documented
rendering-only differences. The default verifier therefore checks that their
pixel-difference counts remain exactly those audited values and reports a
pass; `python reproduce/check_against_paper.py --strict-field-pixels` retains
the zero-pixel audit and reports failure for them.

## Tier B: inversion replay

Tier B uses released checkpoints, truth archives and basis files to replay the
10,000-step exact and surrogate inversions. Historical exact runs took about
1.7--2.2 hours on eight CPU cores; surrogate runs took about 1.8--4.0 minutes
on one NVIDIA L40. These are operational, not hardware-normalised, timings.

CPU calculations are expected to agree to roundoff when the pinned FEM stack
is used. CUDA kernels and parallel PyTorch reductions are not guaranteed to be
bitwise deterministic across GPU models or software builds. Compare reported
metrics with the tolerances in `configs/`, not checkpoint bytes or last-bit
trajectory equality.

## Tier C: data and training replay

Tier C regenerates the Matérn-mixture data, derivative labels and bases and
repeats training. It requires a cluster and substantial storage. The paper
campaign declares 52 training configurations capped at 2,500 epochs; the
downstream paper set freezes 32 selected checkpoints. The production dataset
contains 10,000 training, 2,500 validation and 2,500 independent-test fields,
with 16 derivative directions per field.

Generic scheduler templates are under `cluster/`. They deliberately contain
placeholders rather than a personal account, partition or filesystem path.
The authors ran and approved all production cluster jobs.

## What is and is not bit-for-bit

- Frozen table `.tex` products: byte-for-byte.
- Tier-A plotted arrays: exact SHA-256 match.
- Figures: the required acceptance criterion is zero-pixel difference at 200
  dpi in strict mode. Figures 1, 3 and 5 meet it; Figures 2 and 4 have exact
  plotted data and author-accepted layout/rendering differences described
  above. PDF container hashes can vary with external TeX/font or PDF backends.
- Training and GPU inversion: numerical agreement within documented metric
  tolerances; bitwise identity is not promised.
