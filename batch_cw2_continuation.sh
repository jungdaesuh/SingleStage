#!/bin/bash
#SBATCH -A m4680
#SBATCH --job-name=cw2_cont
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

# Root containing iota_tar* folders (same tree as single_stage_epsilon_constraint.py writes)
SCAN_ROOT="${SCAN_ROOT:-../single_stage_scans_epsilon_constraint_updated/wout_nfp22ginsburg_000_000281_init_dir90}"

# Resume from this stage / resolution (must match your completed cw=1 runs)
STAGE_CW1_REL="${STAGE_CW1_REL:-stage02_cw1/mpol6_ntor6}"

# Extra continuation weight(s), comma-separated
CW_SCHEDULE="${CW_SCHEDULE:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Concurrent jobs: default = floor(cpus / threads); override with MAX_JOBS if unset
if [[ -z "${MAX_JOBS:-}" ]]; then
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    MAX_JOBS=$(( SLURM_CPUS_PER_TASK / OMP_NUM_THREADS ))
  else
    MAX_JOBS=16
  fi
fi

module purge 2>/dev/null || true
if [[ -f /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh ]]; then
  source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
fi
module load python/3.11 2>/dev/null || true
if command -v conda &>/dev/null; then
  conda activate simsopt 2>/dev/null || true
fi

shopt -s nullglob
mapfile -t IOTA_DIRS < <(printf '%s\n' "${SCAN_ROOT}"/iota_tar* | sort -V)
shopt -u nullglob

if [[ ${#IOTA_DIRS[@]} -eq 0 ]]; then
  echo "No iota_tar* directories under: $SCAN_ROOT" >&2
  exit 1
fi

echo "SCAN_ROOT=$SCAN_ROOT"
echo "STAGE_CW1_REL=$STAGE_CW1_REL"
echo "CW_SCHEDULE=$CW_SCHEDULE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MAX_JOBS=$MAX_JOBS"
echo "Found ${#IOTA_DIRS[@]} iota directories."

running=0
completed=0
idx=0

for d in "${IOTA_DIRS[@]}"; do
  name=$(basename "$d")
  iota="${name#iota_tar}"
  src="${d}/${STAGE_CW1_REL}"
  if [[ ! -d "$src" ]]; then
    echo "[skip] $name — missing $STAGE_CW1_REL"
    continue
  fi

  # Basename must start with 90_ for OUT_ROOT (see single_stage_epsilon_constraint.py)
  link="./90_restart_${idx}"
  real_src=$(realpath "$src")

  echo "[launch $((idx + 1))] $name (iota_target=$iota)"

  (
    ln -sfn "$real_src" "$link"
    python3 single_stage_epsilon_constraint.py \
      --init-dir "$link" \
      --iota-target "$iota" \
      --current-weight-schedule "$CW_SCHEDULE"
    status=$?
    rm -f "$link"
    exit "$status"
  ) &

  ((running++)) || true
  ((idx++)) || true

  if (( running >= MAX_JOBS )); then
    wait -n
    ((running--)) || true
    ((completed++)) || true
    echo "[$(date +%H:%M:%S)] Finished one job — completed: $completed, still running: $running"
  fi
done

while (( running > 0 )); do
  wait -n
  ((running--)) || true
  ((completed++)) || true
  echo "[$(date +%H:%M:%S)] Finished one job — completed: $completed, still running: $running"
done

echo "Done. Completed $completed parallel stage(s)."
