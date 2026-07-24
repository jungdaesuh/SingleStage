#!/bin/bash -l
#SBATCH --account=apam
#SBATCH --job-name=IIstage
#SBATCH --partition=apam1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-9%3
#SBATCH --output=logs/stage2_%A_%a.out
#SBATCH --error=logs/stage2_%A_%a.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi

# Thread settings
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

# Make log directory
mkdir -p logs

# TF_a values to run
TF_VALUES=(
    0.500
    0.525
    0.550
    0.575
    0.600
    0.625
    0.650
    0.675
    0.700
    0.725
)

# Select TF_a based on array index
TF_A=${TF_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "====================================="
echo "Running TF_a = ${TF_A}"
echo "Job ID    = ${SLURM_JOB_ID}"
echo "Array ID  = ${SLURM_ARRAY_TASK_ID}"
echo "Node      = $(hostname)"
echo "Start     = $(date)"
echo "====================================="

/burg-archive/home/tg2998/simsopt/venv/bin/python -u \
/burg-archive/home/tg2998/simsopt/examples/dipoles/stage2_to_single_stage.py \
${TF_A}

echo "Finished at $(date)"