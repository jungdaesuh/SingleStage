#!/bin/bash -l
# SLURM array template: one array task = one row of cases.tsv.
# Submit via submit_bridge_cases.sh (it does preflight and sets --array);
# do not sbatch this file directly unless you know the row count.
#
# ===================== EDIT ME (cluster environment) =====================
#SBATCH --account=EDIT_ME_ACCOUNT
#SBATCH --partition=EDIT_ME_PARTITION
#SBATCH --job-name=stage2_bridge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Walltime note: the fixed pipeline converged in 2-3 min at TF_a=0.4 on a
# 32-core workstation. Ginsburg's 24-core walltime is not yet measured, so
# 02:00:00 is deliberately generous; a case that hits it is itself a signal.
# =========================================================================
set -euo pipefail

CASES=${CASES:?submit_bridge_cases.sh exports this}
PYTHON_BIN=${PYTHON_BIN:-/burg-archive/home/tg2998/simsopt/venv/bin/python}
DRIVER_PATH=${DRIVER_PATH:-/burg-archive/home/tg2998/simsopt/examples/dipoles/stage2_to_single_stage.py}
STAGE2_ROOT=${STAGE2_ROOT:?submit_bridge_cases.sh exports this}
OUTPUT_ROOT=${OUTPUT_ROOT:?submit_bridge_cases.sh exports this}
THREADS=${SLURM_CPUS_PER_TASK:?SLURM did not set cpus per task}
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}

row=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CASES}")
[[ -n "${row}" ]] || { echo "no case row ${SLURM_ARRAY_TASK_ID} in ${CASES}"; exit 1; }
IFS=$'\t' read -r -a fields <<<"${row}"
(( ${#fields[@]} == 9 )) \
  || { echo "case row ${SLURM_ARRAY_TASK_ID} is not the validated 9-column schema"; exit 1; }
TF_A=${fields[0]}
POLARITY=${fields[1]}
IOTA_TARGET=${fields[2]}
CURRENT_LIMIT=${fields[3]}
MPOL=${fields[4]}
NTOR=${fields[5]}
MAXITER=${fields[6]}
STEP_RADIUS=${fields[7]}
VOLUME_TARGET=${fields[8]}

args=(
  "${TF_A}"
  --field-polarity "${POLARITY}"
  --iota-target "${IOTA_TARGET}"
  --current-limit-amperes "${CURRENT_LIMIT}"
  --mpol "${MPOL}"
  --ntor "${NTOR}"
  --maxiter "${MAXITER}"
  --outer-step-radius "${STEP_RADIUS}"
)
[[ "${VOLUME_TARGET}" != "-" ]] && args+=(--volume-target "${VOLUME_TARGET}")
args+=(--stage2-root "${STAGE2_ROOT}" --output-root "${OUTPUT_ROOT}")

echo "case ${SLURM_ARRAY_TASK_ID}: ${DRIVER_PATH} ${args[*]}"
exec srun --cpu-bind=cores "${PYTHON_BIN}" -u "${DRIVER_PATH}" "${args[@]}"
