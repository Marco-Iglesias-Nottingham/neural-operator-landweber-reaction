#!/bin/bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN="${ROOT}/configs/paper/training_campaign.tsv"

[[ -f "${CAMPAIGN}" ]]

awk -F'\t' '
NR == 1 {
    for (i = 1; i <= NF; ++i) column[$i] = i
    required[1] = "run_id"
    required[2] = "model_family"
    required[3] = "training_epochs"
    for (j = 1; j <= 3; ++j) {
        if (!(required[j] in column)) {
            printf "ERROR: missing campaign column %s\n", required[j] > "/dev/stderr"
            exit 1
        }
    }
    next
}
{
    rows++
    run_id = $column["run_id"]
    family = $column["model_family"]
    epochs = $column["training_epochs"] + 0
    if (seen[run_id]++) {
        printf "ERROR: duplicate run_id %s\n", run_id > "/dev/stderr"
        exit 1
    }
    if (!(family == "fno" || family == "deeponet" || family == "genfno" || family == "gendeeponet")) {
        printf "ERROR: unsupported model family %s\n", family > "/dev/stderr"
        exit 1
    }
    if (epochs < 1 || epochs > 2500) {
        printf "ERROR: invalid epoch budget %s for %s\n", epochs, run_id > "/dev/stderr"
        exit 1
    }
}
END {
    if (rows != 52) {
        printf "ERROR: expected 52 declared configurations, found %d\n", rows > "/dev/stderr"
        exit 1
    }
}
' "${CAMPAIGN}"

for run_id in \
    fno_N10000_w24_m10_L4_J300 \
    do_N10000_Dp400_L6_J100 \
    gfno_N10000_im16_om24_h512_rb4_J100 \
    gdo_N10000_M400_N300_h1024_rb4_J1000
do
    awk -F'\t' -v target="${run_id}" 'NR > 1 && $1 == target { found = 1 } END { exit !found }' "${CAMPAIGN}"
done

if awk -F'\t' 'NR > 1 && $1 == "fno_N5000_w8_m4_L4_J10" { found = 1 } END { exit !found }' "${CAMPAIGN}"; then
    echo 'ERROR: excluded cross-grid configuration is present.' >&2
    exit 1
fi

echo 'Public training-campaign contract checks passed.'
