# Neural-operator Landweber iteration: Reaction example

This repository contains the code and frozen numerical products for the
Reaction example in *Neural Operator Surrogates for Landweber Iteration in
Solving Inverse Problems* by Kuo Gai, Marco A. Iglesias, Shuai Lu and Frank
Werner.
It implements the elliptic reaction-coefficient forward problem, derivative
labels, four neural-operator architectures, covariance-preconditioned
Landweber iteration, and the analyses reported in Sections 7.1--7.11.

The manuscript itself is not part of this repository. The public release is
prepared from a frozen 28 October 2026 paper snapshot; the twelve files under
`reference/paper/` are the corresponding Reaction tables and figures used for
automated verification.

## Quick start: Tier A

Create the small product environment, install the Zenodo bundle, and rebuild
the paper products:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r environments/tier-a-requirements.txt
python reproduce/download_data.py
python reproduce/paper.py
```

The frozen data and numerical artifacts are archived at
[doi:10.5281/zenodo.23767592](https://doi.org/10.5281/zenodo.23767592).
The published record is the data companion to software release `v1.0.1`.
Alternatively, a local archive can be installed with
`python reproduce/download_data.py --archive /path/to/reaction-paper-products-v1.zip`,
or an unpacked bundle with `--source /path/to/reaction-paper-v1`.
`reproduce/paper.py` verifies the bundle, regenerates seven tables and five
figures, checks the numerical claims quoted only in the text, and compares the
outputs with the frozen references.

The data, tables, prose claims and three compact figures pass their frozen
checks. The two field figures use the exact frozen arrays, ordering, values
and colour limits; their portable rendering has the documented text/layout
differences described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The default
verification accepts those two author-approved rendering-only differences;
`check_against_paper.py --strict-field-pixels` retains the zero-pixel audit.

## Reproduction tiers

- **Tier A (minutes, CPU):** regenerate all paper products from checksummed
  saved results. This is the fully tested route.
- **Tier B (GPU hours):** rerun inversions from frozen checkpoints and data.
  Results may differ in last digits because of GPU reduction nondeterminism.
- **Tier C (cluster):** regenerate training data, bases and derivative labels,
  then retrain. Generic scheduler templates are in `cluster/`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for scope and run-time estimates.
For a stage-by-stage map from data generation to the final manuscript
products, see [docs/pipeline.md](docs/pipeline.md).

## Repository layout

| Path | Purpose |
|---|---|
| `reaction_core/` | FEM, grids, random fields, covariance actions and inversion utilities |
| `models/` | FNO, DeepONet, GenFNO and GenDeepONet implementations |
| `scripts/` | Data generation, training, evaluation, inversion and reported analyses |
| `configs/` | Frozen paper campaign, truth and inversion selections |
| `reproduce/` | Small Tier-A paper-product interface |
| `tests/` | Unit and contract tests |
| `cluster/` | Generic Tier-B/Tier-C Slurm templates |
| `reference/paper/` | Frozen Reaction outputs used only for verification |

The authoritative entry points and the artifacts passed between these
directories are documented in [docs/pipeline.md](docs/pipeline.md).

## Paper-product map

| Paper item | Output | Command | Tier-A source |
|---|---|---|---|
| Table 1 | `headline_emulator_test.tex` | `make_tables.py` | `tables/headline_emulator_test.csv` |
| Table 2 | `noiseless_headline_T1_T2.tex` | `make_tables.py` | `tables/noiseless_headline_T1_T2.csv` |
| Table 3 | `fno_capacity_semiconvergence.tex` | `make_tables.py` | `tables/fno_capacity_semiconvergence.csv` |
| Table 4 | `fno_training_size_semiconvergence.tex` | `make_tables.py` | `tables/fno_training_size_semiconvergence.csv` |
| Table 5 | `preconditioner_low_smoothness_errors.tex` | `make_tables.py` | `revision/E3/*.csv` |
| Table 6 | `table_low_smoothness_diagnostics_single.tex` | `make_tables.py` | `revision/E1_E2/trajectory_diagnostics_summary.csv` |
| Table 7 | `noisy_tau_stopping_grouped.tex` | `make_tables.py` | `revision/E5/noisy_tau_stopping.csv` |
| Figure 1 | `noiseless_convergence_compact.pdf` | `make_figures.py` | noiseless convergence CSVs |
| Figure 2 | `noiseless_oracle_reconstructions_rows.pdf` | `make_figures.py` | `reconstruction_fields.npz` |
| Figure 3 | `derivative_geometry_compact.pdf` | `make_figures.py` | `derivative_geometry.csv` |
| Figure 4 | `preconditioner_identity_fields_T2.pdf` | `make_figures.py` | `reconstruction_fields.npz` |
| Figure 5 | `noisy_convergence_T1_compact.pdf` | `make_figures.py` | noisy T1 convergence CSV |

All Tier-A source paths are relative to `data/reaction-paper-v1/`. The data
root can be changed with `REACTION_DATA_ROOT`.

## Historical hardware

Exact FEM inversions used eight CPU cores on Sulis. Surrogate inversions and
training used one NVIDIA GPU (L40 for the reported inversion timing; A100 for
the selected training campaigns). The emulator stack was Python 3.11,
PyTorch 2.4.0 and CUDA 12.4; the reproduction FEM stack is DOLFINx 0.10.0 and
PETSc 3.25.0. See `docs/timing.md` and `environments/`.

## Related repository

The Transient-Darcy application in the same paper is maintained as a separate
repository. Its public URL will be inserted when both releases are approved.

## Citation, licence and contact

Citation metadata are in [CITATION.cff](CITATION.cff). The software-licence
is BSD-3-Clause; the separately deposited data and numerical artifacts are
CC BY 4.0. The public source repository is
[Marco-Iglesias-Nottingham/neural-operator-landweber-reaction](https://github.com/Marco-Iglesias-Nottingham/neural-operator-landweber-reaction).
Correspondence: Marco A. Iglesias.
