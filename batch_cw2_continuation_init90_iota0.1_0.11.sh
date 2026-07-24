#!/bin/bash
# Modified cw=2 continuation: only iota_tar0.1 and iota_tar0.11 under init_dir 90
# (linspace 0.10 is stored as folder iota_tar0.1). Same mechanics as batch_cw2_continuation.sh.
#
#SBATCH -A m4680
#SBATCH --job-name=cw2_i01
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

SCAN_ROOT="${SCAN_ROOT:-../single_stage_scans_epsilon_constraint_updated/wout_nfp22ginsburg_000_000281_init_dir90}"
STAGE_CW1_REL="${STAGE_CW1_REL:-stage02_cw1/mpol6_ntor6}"
CW_SCHEDULE="${CW_SCHEDULE:-2}"

export OMP_NUM_THREADS=64

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

# Only these two iota targets (0.10 -> iota_tar0.1 in directory names)
IOTA_DIRS=(
  "${SCAN_ROOT}/iota_tar0.1"
  "${SCAN_ROOT}/iota_tar0.11"
)

echo "SCAN_ROOT=$SCAN_ROOT"
echo "STAGE_CW1_REL=$STAGE_CW1_REL"
echo "CW_SCHEDULE=$CW_SCHEDULE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MAX_JOBS=$MAX_JOBS"
echo "Restricted to ${#IOTA_DIRS[@]} iota directories (init_dir 90, iota 0.10 and 0.11)."

running=0
completed=0
idx=0

for d in "${IOTA_DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "[skip] missing directory: $d" >&2
    continue
  fi
  name=$(basename "$d")
  iota="${name#iota_tar}"
  src="${d}/${STAGE_CW1_REL}"
  if [[ ! -d "$src" ]]; then
    echo "[skip] $name — missing $STAGE_CW1_REL"
    continue
  fi

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
