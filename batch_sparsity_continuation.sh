#!/bin/bash
#SBATCH -A m4680
#SBATCH --job-name=sparse_cont
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

# Purge all modules to remove the "Application linked against multiple cray-libsci libraries" warning
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh

# Load Python module
module load python/3.11

# Activate conda environment
conda activate simsopt

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
# Scan root under ../single_stage_scans_epsilon_constraint_updated/
SCAN_ROOT="../single_stage_scans_epsilon_constraint_updated/wout_nfp22ginsburg_000_000281_init_dir90"

# Source stage directory and iteration sub-directory to initialize from.
SOURCE_STAGE="stage00_cw0.25"
ITER_DIR="mpol6_ntor6"

# Current-weight continuation schedule for the sparse optimization.
CW_SCHEDULE="0.5,1,2"

# Iota targets to process (one Python process per iota, parallelized below).
# Edit to match the iota_tar* directories you want.
IOTA_TARGETS=(
  0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25
)

# Threading: each Python process gets OMP_NUM_THREADS threads.
export OMP_NUM_THREADS=8
MAX_JOBS=$(( SLURM_CPUS_PER_TASK / OMP_NUM_THREADS ))
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
fi

echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS per job, MAX_JOBS=$MAX_JOBS"
echo "Scan root: $SCAN_ROOT"
echo "Source stage: $SOURCE_STAGE/$ITER_DIR"
echo "Current-weight schedule: $CW_SCHEDULE"
echo "Iota targets to process: ${#IOTA_TARGETS[@]}"

running=0
completed=0
total=${#IOTA_TARGETS[@]}

for iota in "${IOTA_TARGETS[@]}"; do
  INIT_DIR="${SCAN_ROOT}/iota_tar${iota}/${SOURCE_STAGE}/${ITER_DIR}"

  if [[ ! -d "$INIT_DIR" ]]; then
    echo "[$((completed + running + 1))/$total] Skipping iota=$iota: $INIT_DIR not found"
    ((completed++))
    continue
  fi

  echo "[$((completed + running + 1))/$total] Starting sparse continuation for iota=$iota"

  python3 single_stage_epsilon_constraint.py \
    --init-dir "$INIT_DIR" \
    --sparse \
    --iota-target "$iota" \
    --current-weight-schedule "$CW_SCHEDULE" &
  ((running++))

  # If we've reached the concurrency limit, wait for one job to finish.
  if (( running >= MAX_JOBS )); then
    wait -n
    ((running--))
    ((completed++))
    echo "[$(date +%H:%M:%S)] One iota finished — completed: $completed/$total, still running: $running"
  fi
done

# Wait for any remaining jobs to finish.
while (( running > 0 )); do
  wait -n
  ((running--))
  ((completed++))
  echo "[$(date +%H:%M:%S)] One iota finished — completed: $completed/$total, still running: $running"
done

echo "All $total sparse continuation runs completed."
