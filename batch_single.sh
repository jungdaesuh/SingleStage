#!/bin/bash
#SBATCH --account=m4680
#SBATCH --job-name=single_stage
#SBATCH --time=02:00:00
#SBATCH -C cpu
#SBATCH -o ../single_stage_outputs/slurm_outputs/job_output_%j.out
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL

# Load Python module
module load python/3.11

# Activate virtual environment
conda activate simsopt

# Run the simulation
python3 single_stage_dipole_example.py \
  --iota-target 0.08 \
  --iota-weight 100 \
  --res-weight 100 \
  --current-threshold 190000 \
  --current-weight 10 \
