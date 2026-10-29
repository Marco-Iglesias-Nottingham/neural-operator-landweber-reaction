# Reaction numerical pipeline

This page is the top-level map from the mathematical experiment to the code
and frozen artifacts. It describes the pipeline used for the paper, rather
than every historical experiment retained in the private research repository.
The authoritative campaign membership is defined by the files in
`configs/paper/`; filenames or modification dates are not used to decide which
runs belong to the paper.

## Pipeline at a glance

```text
frozen distribution and grids
        |
        +--> derivative-direction basis
        |
        +--> train / validation / independent-test FEM shards
                      |
                      +--> input-PCA and output-POD bases
                      |
                      +--> train 52 declared configurations
                                  |
                                  +--> validation-only model selection
                                              |
                                              +--> 32 frozen checkpoints
                                                        |
frozen T1/T2 truths and noisy data ---------------------+
                                                        |
                                                        +--> exact and surrogate
                                                             Landweber trajectories
                                                                    |
independent test evaluation ----------------------------------------+
                                                                    |
                                                                    +--> paper analyses
                                                                         and Tier-A data
                                                                                |
                                                                                +--> tables,
                                                                                     figures,
                                                                                     text checks
```

The final-test set is not used in training or checkpoint selection. The paper
tables and figures are generated from saved, checksummed numerical summaries;
they are not the primary scientific data.

## Frozen configuration files

| File | Contract |
|---|---|
| `configs/paper/training_campaign.tsv` | The 52 declared training configurations, each capped at 2,500 epochs |
| `configs/paper/downstream_models.tsv` | The 32 validation-selected checkpoints used downstream: 4 derivative-informed headline models, 4 forward-only headline models and 24 controlled ablations |
| `configs/paper/truth_definitions.tsv` | The frozen T1 held-out draw and structured T2 truth |
| `configs/paper/inversion_matrix.tsv` | The required noiseless and 1%, 2% and 5% noisy T1/T2 inversion cases |
| `configs/revision_2026_09/geometry_sweep.tsv` | The low-smoothness Matérn geometry sweep |

These files should be read before launching a campaign. A regenerated
artifact is part of the paper pipeline only if it satisfies these contracts.

## Stage-by-stage code map

### 1. Derivative-direction basis

`scripts/data/build_kl_basis.py` samples the same broad non-periodic Matérn
mixture used by the production data and constructs the 16 covariance-scaled
physical perturbations used for derivative labels. The saved basis is an
input to every production shard. See `docs/derivative_directions.md` for the
precise interpretation of these directions.

### 2. Training, validation and independent-test data

`scripts/data/generate_shard.py` generates one HDF5 shard. For every sampled
coefficient it stores the physical coefficient field, the exact FEM forward
solution and 16 exact tangent/JVP labels. Independent one-rank jobs generate
non-overlapping shards.

The frozen split is:

| Split | Shards | Fields | Use |
|---|---:|---:|---|
| training | 16 | 10,000 | optimisation and construction of reduced bases |
| validation | 4 | 2,500 | checkpoint and hyperparameter selection |
| independent test | 4 | 2,500 | final `E_F` and `E_J` reporting only |

Each production shard contains 625 fields. The split, seeds, offsets,
distribution metadata and basis identity must be checked from the HDF5
metadata rather than inferred from the directory name.

### 3. Reduced input and output bases

`scripts/data/build_weighted_basis.py` builds the quadrature-weighted input
PCA and output POD bases used by GenDeepONet.
The paper campaign uses frozen bases for the 1,000-, 5,000- and 10,000-sample
training subsets. These basis files are model inputs and therefore belong with
the released checkpoints, not with generated paper figures.

### 4. Model training

The architecture-specific entry points are:

- `scripts/training/train_fno.py`;
- `scripts/training/train_deeponet.py`;
- `scripts/training/train_genfno.py`; and
- `scripts/training/train_gendeeponet.py`.

Shared loss, data-loading, normalisation and checkpoint logic is in
`scripts/training/training_utils.py`. The declared configurations are in
`configs/paper/training_campaign.tsv`. Training writes epoch checkpoints and
validation histories, but downstream analyses use only the 32 checkpoints
listed in `configs/paper/downstream_models.tsv`.

### 5. Validation selection and independent testing

`scripts/evaluation/summarize_validation.py` collects validation metrics and
`scripts/evaluation/select_inversion_models.py` applies the validation-only
selection rule. It must reproduce `configs/paper/downstream_models.tsv`.

`scripts/evaluation/evaluate_checkpoints.py` evaluates frozen checkpoints on
the reserved test shards. Forward error uses all 2,500 test inputs. The
reported derivative test uses 256 inputs and all 16 frozen directions.
`scripts/evaluation/collect_final_test.py` assembles the machine-readable
summaries. Test results do not feed back into checkpoint selection.

### 6. Frozen inverse-problem truths

`scripts/data/generate_truth_T1.py` creates the held-out draw from the training
distribution, and `scripts/data/generate_truth_T2.py` creates the structured
two-inclusion truth. Both use the fine FEM writer in
`scripts/data/generate_truth.py`. `scripts/data/create_noisy_truth.py` adds the
paired 1%, 2% and 5% noise realisations. Exact definitions and seeds are in
`configs/paper/truth_definitions.tsv`.

### 7. Landweber inversion

The two production drivers are:

- `scripts/inversion/landweber_exact.py` for the exact FEM forward map; and
- `scripts/inversion/landweber_surrogate.py` for a frozen neural operator.

They share the 120 by 120 parameter grid, initial field `m_0=100`, admissible
box `[0,200]`, adaptive step rule, physical-change cap and 10,000-iteration
paper budget. The baseline geometry is the non-periodic Matérn covariance
with `nu_C=3/2` and `ell_C=0.1`; identity and low-smoothness geometries are
controlled comparison arms. Noisy trajectories are run for the full budget,
and discrepancy crossings are evaluated retrospectively. See
`docs/inversion.md` for the discretisation and geometry conventions.

`configs/paper/inversion_matrix.tsv` is the compact statement of which T1/T2,
noise and model cohorts are required. Generic CPU and GPU scheduler templates
are in `cluster/`; they deliberately require site-specific paths and resource
names to be supplied by the user.

### 8. Analysis and paper products

The reported analysis scripts are the selected files under
`scripts/revision_2026_09/` and `scripts/revision_2026_10/`. They reduce the
saved test and inversion outputs to the CSV and NPZ files in the Zenodo
Tier-A bundle. `reproduce/paper.py` is the public entry point that verifies
that bundle and regenerates the seven tables, five figures and numerical
claims used in the manuscript.

The generated `.tex`, PDF and PNG files are outputs, not required input
artifacts. The frozen reference copies under `reference/paper/` are retained
only for automated comparison.

## Reproduction routes

| Route | Starts from | What it establishes |
|---|---|---|
| Tier A: paper products | checksummed summary bundle | Recreates all reported Reaction tables, figures and in-text numerical checks on a laptop |
| Tier B: inversion replay | frozen checkpoints, bases and truth archives | Repeats exact and surrogate Landweber trajectories and then the analyses |
| Tier C: training replay | FEM shards, derivative basis and frozen configs | Rebuilds reduced bases, retrains models, repeats selection and independent testing before Tier B |

Tier A is the normal paper-reproduction route. Tier B is needed to inspect or
modify inversion trajectories. Tier C is the costly end-to-end scientific
reproduction route.

## What is outside this pipeline

Historical post-2,500-epoch continuations, DIS retraining, projection/clipping
diagnostics, smoke tests and superseded launchers are retained privately for
provenance. They are not inputs to the frozen paper products and should not be
mixed into the public pipeline merely because their source files exist in the
private repository.

## Cluster launchers and account privacy

The production `slurm/` tree from the private research repository is not
exported. It contains site-specific assumptions and historical provenance.
The public repository instead contains three generic templates under
`cluster/` for training, exact-FEM inversion and surrogate inversion. Their
account, partition, GPU type, environment initialisation and repository path
are placeholders.

After cloning the public repository on a cluster, copy and customise those
templates under `local_cluster/`. That directory is ignored by Git and must
remain local to the cluster. Scheduler output and large result directories are
also ignored. Before any public commit, the staged tree is checked for the
private username, home/GPFS paths, login hostname, account directives, SSH-key
paths, job logs and private manuscript paths. The cluster name may appear in
historical hardware documentation, but no credential or account identifier
is required to use the public code.

## Artifact boundary

The Git repository contains source code, frozen configurations, small tests
and generic scheduler templates. Large binary artifacts belong in Zenodo:

1. the small Tier-A paper-product source bundle;
2. the frozen checkpoints, bases and T1/T2 truth archives needed for Tier B;
3. the 16 training, 4 validation and 4 independent-test FEM shards needed for
   Tier C.

The exact proposed deposit layout and the distinction between the minimum
headline checkpoint set and the complete 32-checkpoint paper set are given in
`DATA.md`.
