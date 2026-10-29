# Generic cluster templates

These templates document Tier-B/Tier-C job structure. Replace every value in
angle brackets for the target cluster and review resource requests before
submission. They contain no Sulis account, username or personal path.

The templates are intentionally not a one-command production campaign: data
and checkpoint locations must be provided explicitly, and the preflight mode
of the underlying scripts should be used before a full submission.

Do not edit these tracked templates with a personal account, username,
partition or filesystem path. On a cluster, copy the needed template into the
root-level `local_cluster/` directory and customise that copy. `local_cluster/`,
scheduler logs, results, checkpoints and data are ignored by Git.

The original production `slurm/` directory is private provenance and is not
part of this release. Before publishing, the release staging tree is scanned
for personal usernames, login hosts, home/GPFS paths, account directives,
SSH-key paths and raw scheduler logs.
