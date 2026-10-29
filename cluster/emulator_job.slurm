#!/bin/bash
#SBATCH --job-name=reaction_emulator
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:<gpu-type>:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
set -euo pipefail
source <conda-initialisation>
conda activate reaction-emulator
cd <repository-root>
srun python scripts/inversion/landweber_surrogate.py "$@"
