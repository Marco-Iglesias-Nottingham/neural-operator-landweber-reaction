# Data and artifact layout

Large scientific artifacts are deposited separately from the Git repository.
The frozen record is
[doi:10.5281/zenodo.23767592](https://doi.org/10.5281/zenodo.23767592),
version 1.0.0. It contains nine ZIP archives plus a README, data licence,
master manifest and master SHA-256 list. The archives preserve the logical
project paths and may be extracted into the same parent directory.

| Zenodo file | Contents |
|---|---|
| `reaction-training-shards-part1-v1.zip`--`part4-v1.zip` | 16 training shards, four complete shards per archive |
| `reaction-validation-and-reduced-bases-v1.zip` | 4 validation shards and the six input-PCA/output-POD bases |
| `reaction-final-test-and-kl-basis-v1.zip` | 4 independent-test shards and the frozen 16-direction basis |
| `reaction-frozen-models-v1.zip` | 32 selected checkpoints and frozen registries |
| `reaction-frozen-truths-v1.zip` | clean and 1%, 2%, 5% noisy T1/T2 truths and metadata |
| `reaction-paper-products-v1.zip` | checksummed machine-readable inputs for all reported Reaction products |

The complete payload contains 101 files and 17,612,852,162 bytes before ZIP
container overhead. The training set is split only for upload reliability;
no HDF5 shard is divided between archives.

## Group A: paper-product sources (required, small)

The Reaction-specific bundle `reaction-paper-v1` contains only the saved
numerical products required by the paper-product scripts.

| Directory | Contents |
|---|---|
| `tables/` | Independent-test, inversion and supplementary numerical summaries |
| `figure_data/` | Complete plotted trajectories and extracted displayed fields |
| `revision/E1_E2/` | Trajectory diagnostics for the low-smoothness sweep |
| `revision/E3/` | Geometry-sweep outcomes and reference arms |
| `revision/E5/` | Noisy discrepancy-threshold analysis |
| `revision/E8/`, `revision/E9/` | Derivative-geometry and cost accounting |
| `manifests/` | Original paper-product provenance |
| `SHA256SUMS` | SHA-256 for every payload file |

`manifest.json` is the authoritative, machine-readable file list. It records
the relative path, exact byte size, SHA-256 digest, internal source-relative
provenance path and source digest for each of the 22 payload files. It contains
no personal or cluster filesystem paths. `download_data.py`
checks both the manifest and `SHA256SUMS` before installing the bundle.

This group is sufficient to regenerate the published tables, figures and
computation-derived prose claims. Generated `.tex`, PDF and PNG files do not
need to be deposited as scientific inputs; the repository can recreate them.
The frozen rendered references remain in Git solely for automated checks.

## Group B: frozen inversion inputs (required for Tier B)

This group should contain:

- the 32 checkpoints listed in `configs/paper/downstream_models.tsv`;
- the frozen 16-direction derivative/KL basis;
- the input-PCA and output-POD basis files required by the 1,000-, 5,000- and
  10,000-sample reduced models;
- the clean T1 and T2 truth archives; and
- the paired T1/T2 1%, 2% and 5% noisy truth archives.

If a smaller headline-only file is desirable, it must contain eight
checkpoints: four derivative-informed headline models and their four
forward-only counterparts. Four derivative-informed headline checkpoints
alone cannot reproduce the paper's derivative-training comparison. The
remaining 24 checkpoints can be a separate `extended-ablations` download, but
all 32 are required to replay every reported downstream model arm.

Every checkpoint and basis should be listed in a TSV/JSON index giving its
run ID, role, epoch, architecture, required basis files, byte size and SHA-256
digest. Training histories are useful provenance but are not required for
Tier B when the frozen checkpoint choice and validation metrics are included.

## Group C: frozen FEM dataset (required for Tier C)

This group should contain exactly the production split:

- 16 training shards (10,000 fields);
- 4 validation shards (2,500 fields); and
- 4 independent-test shards (2,500 fields).

The HDF5 shards already contain coefficient fields, forward solutions and 16
exact JVP labels, so no separate derivative-label download is needed. Include
the shard logs only if they contain useful non-sensitive metadata not already
stored in HDF5; otherwise a clean manifest with seeds, offsets, sample counts,
basis hash, sizes and SHA-256 digests is preferable.

## What does not need to be deposited

The minimum reproducibility deposit does not need:

- generated tables, figures or intermediate plotting directories;
- all epoch checkpoints or post-2,500-epoch continuations;
- private DIS experiments and their derivative sidecars;
- duplicate Sulis snapshots or full working-directory backups;
- scheduler stdout/stderr containing personal paths; or
- full saved-state trajectory archives when their reported summaries are
  already in Group A and the trajectories can be rerun from Group B.

Selected run metadata and environment captures should be sanitised and kept
in Git or the artifact manifest. Full trajectories may be offered as an
optional provenance download, but they are not necessary for reproducing the
published products.

## Deposit metadata and licences

Each downloadable group has its own checksum manifest and README, and the
Zenodo record links to the exact Git commit and software release tag. The
software version is `v1.0.0`; the data-record version is `1.0.0`. Source code
is BSD-3-Clause and deposited data and numerical artifacts are CC BY 4.0.

The Group-A archive is about 48 MB. The remaining archives are separated so a
Tier-A reader does not need to download the full training corpus. Exact sizes
and SHA-256 digests are in the Zenodo master manifest.

The Zenodo record was created by the author and assigned the reserved DOI
above. Publication remains an author action after the GitHub URL and final
metadata have been checked.
