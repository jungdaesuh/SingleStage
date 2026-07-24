#!/bin/bash
#SBATCH -A m4680
#SBATCH --job-name=add_sparsity
#SBATCH --time=12:00:00
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
# Init-dir name under ../single_stage_scans_epsilon_constraint_updated/ (or pass a full path).
# Legacy: ../single_stage_scans_no_sparsity_epsilon_constraint/ is also tried by the Python script.
#INIT_DIR_ROOT="wout_nfp22ginsburg_000_000281_init_dir91"
#INIT_DIR_ROOT="wout_nfp22ginsburg_000_001242_init_dir01"
#INIT_DIR_ROOT="wout_nfp22ginsburg_000_001242_init_dir08"
INIT_DIR_ROOT="wout_nfp22ginsburg_000_000281_init_dir90"

# Boozer volume branch for iota_tar* vs iota_tar*_vol0.35 / _vol0.4. Use 0.3 for plain iota_tar*
# (default volume in single_stage_epsilon_constraint). Leave empty only if each iota has a single folder.
VOL_TARGET=0.3

# Run one iota target per Python process (parallelized below).
# Edit to match the iota_tar* directories you want (e.g. ls ../single_stage_scans_epsilon_constraint_updated/$INIT_DIR_ROOT).
IOTA_TARGETS=(
  0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25
)

# Maximum number of optimizer iterations (single_stage_epsilon_constraint.py uses 150 per stage)
MAXITER=200

# Optional source stage index to initialize from (e.g. 3 => stage03_*).
# Leave empty to use the highest non-sparsity stage.
SOURCE_STAGE="3"

# Optional current-weight override for the new sparsity run (e.g. 1).
# Leave empty to inherit from the source stage.
CURRENT_WEIGHT="2.0"

# Threading: each Python process gets OMP_NUM_THREADS threads.
export OMP_NUM_THREADS=8
MAX_JOBS=$(( SLURM_CPUS_PER_TASK / OMP_NUM_THREADS ))
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
fi

echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS per job, MAX_JOBS=$MAX_JOBS"
echo "Init root: $INIT_DIR_ROOT"
echo "Iota targets to process: ${#IOTA_TARGETS[@]}"

running=0
completed=0
total=${#IOTA_TARGETS[@]}

for iota in "${IOTA_TARGETS[@]}"; do
  echo "[$((completed + running + 1))/$total] Starting sparsity continuation for iota=$iota"

  extra=()
  if [[ -n "${VOL_TARGET:-}" ]]; then
    extra+=(--vol-target "$VOL_TARGET")
  fi
  if [[ -n "${SOURCE_STAGE:-}" ]]; then
    extra+=(--source-stage "$SOURCE_STAGE")
  fi
  if [[ -n "${CURRENT_WEIGHT:-}" ]]; then
    extra+=(--current-weight "$CURRENT_WEIGHT")
  fi

  python3 single_stage_add_sparsity.py \
    --init-dir-root "$INIT_DIR_ROOT" \
    --iota-target "$iota" \
    --maxiter "$MAXITER" \
    "${extra[@]}" &
  ((running++))

  # If we've reached the concurrency limit, wait for one iota-run to finish.
  if (( running >= MAX_JOBS )); then
    wait -n
    ((running--))
    ((completed++))
    echo "[$(date +%H:%M:%S)] One iota finished — completed: $completed/$total, still running: $running"
  fi
done

# Wait for any remaining iota-runs to finish.
while (( running > 0 )); do
  wait -n
  ((running--))
  ((completed++))
  echo "[$(date +%H:%M:%S)] One iota finished — completed: $completed/$total, still running: $running"
done

echo "All $total sparsity iota-runs completed."
