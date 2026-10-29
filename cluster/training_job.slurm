#!/bin/bash
#SBATCH --job-name=reaction_train
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:<gpu-type>:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
set -euo pipefail
source <conda-initialisation>
conda activate reaction-emulator
cd <repository-root>
# Choose the architecture-specific trainer and pass the frozen config.
srun python <trainer-from-scripts/training> "$@"
