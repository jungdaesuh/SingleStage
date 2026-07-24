#!/bin/bash
# Rerun single_stage_epsilon_constraint.py at higher Boozer resolution for every
# iota_tar* directory under a scan root (skips directory names containing _vol),
# warm-starting from an existing stage folder (e.g. mpol6_ntor6). Output goes
# next to the source run:
#   <scan>/iota_tar*/<STAGE>/mpol9_ntor9/
#
# Usage (from examples/dipoles, or submit as a Slurm job):
#   ./batch_highres_from_stage.sh
#
# Edit USER SETTINGS below. Current weight in --current-weight-schedule is taken
# from the stage folder name (e.g. stage00_cw0.25 -> 0.25) so only that stage
# tag is written under mpol9_ntor9.

#SBATCH -A m4680
#SBATCH --job-name=ss_highres
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
SCAN_ROOT="../single_stage_scans_epsilon_constraint_updated/wout_nfp22ginsburg_000_000281_init_dir90"

# Stage folder to initialize from (must contain mpol6_ntor6 or edit SOURCE_RES).
SOURCE_STAGE="stage03_cw2"

# Source Boozer resolution subdirectory (coils + surf to load).
SOURCE_RES="mpol6_ntor6"

# Target resolution and optimizer cap.
MPOL=9
NTOR=9
MAXITER=1

# Optional: override current-weight schedule (comma-separated). If empty, parsed
# from SOURCE_STAGE (e.g. stage00_cw0.25 -> 0.25).
CW_SCHEDULE=""

# Threading for each Python process
export OMP_NUM_THREADS=8
_SLURM_CPUS="${SLURM_CPUS_PER_TASK:-8}"
MAX_JOBS=$(( _SLURM_CPUS / OMP_NUM_THREADS ))
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
fi

# -----------------------------------------------------------------------------
# Environment (Slurm / interactive)
# -----------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  module purge
  # shellcheck disable=SC1091
  source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
  module load python/3.11
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate simsopt
fi

if [[ -z "$CW_SCHEDULE" ]]; then
  if [[ "$SOURCE_STAGE" =~ cw([0-9.]+) ]]; then
    CW_SCHEDULE="${BASH_REMATCH[1]}"
  else
    echo "Could not parse current weight from SOURCE_STAGE=$SOURCE_STAGE; set CW_SCHEDULE explicitly."
    exit 1
  fi
fi

echo "Scan root:         $SCAN_ROOT"
echo "Source stage:      $SOURCE_STAGE/$SOURCE_RES"
echo "Target resolution: mpol${MPOL}_ntor${NTOR}, maxiter=$MAXITER"
echo "Current weights:   $CW_SCHEDULE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MAX_JOBS=$MAX_JOBS"

shopt -s nullglob
IOTA_DIRS=()
for _d in "$SCAN_ROOT"/iota_tar*/; do
  [[ -d "$_d" ]] || continue
  _base="$(basename "${_d%/}")"
  # Skip non-default volume scans (iota_tar*_vol*).
  if [[ "$_base" == *_vol* ]]; then
    continue
  fi
  IOTA_DIRS+=("$_d")
done
if (( ${#IOTA_DIRS[@]} == 0 )); then
  echo "No iota_tar* directories under $SCAN_ROOT (after excluding *_vol*)"
  exit 1
fi

running=0
completed=0
total=${#IOTA_DIRS[@]}

for iota_dir in "${IOTA_DIRS[@]}"; do
  iota_dir="${iota_dir%/}"
  INIT_DIR="${iota_dir}/${SOURCE_STAGE}/${SOURCE_RES}"

  if [[ ! -d "$INIT_DIR" ]]; then
    echo "[$((completed + running + 1))/$total] Skip $(basename "$iota_dir"): no $INIT_DIR"
    ((completed++)) || true
    continue
  fi
  if [[ ! -f "$INIT_DIR/results.json" ]]; then
    echo "[$((completed + running + 1))/$total] Skip $(basename "$iota_dir"): missing results.json"
    ((completed++)) || true
    continue
  fi

  IOTA_TARGET="$(python3 -c "import json; d=json.load(open(r'''$INIT_DIR/results.json''')); print(d['graph']['iota_target'])")"

  echo "[$((completed + running + 1))/$total] Starting $(basename "$iota_dir")  iota_target=$IOTA_TARGET  init=$INIT_DIR"

  python3 single_stage_epsilon_constraint.py \
    --init-dir "$INIT_DIR" \
    --iota-target "$IOTA_TARGET" \
    --mpol "$MPOL" \
    --ntor "$NTOR" \
    --maxiter "$MAXITER" \
    --current-weight-schedule "$CW_SCHEDULE" &
  ((running++)) || true

  if (( running >= MAX_JOBS )); then
    wait -n
    ((running--)) || true
    ((completed++)) || true
    echo "[$(date +%H:%M:%S)] One run finished — completed: $completed/$total, running: $running"
  fi
done

while (( running > 0 )); do
  wait -n
  ((running--)) || true
  ((completed++)) || true
  echo "[$(date +%H:%M:%S)] One run finished — completed: $completed/$total, running: $running"
done

echo "All high-resolution runs finished ($total iota_tar directories, excluding *_vol*)."
