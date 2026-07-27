#!/bin/bash -l
#SBATCH --account=seasdean        
#SBATCH --job-name=tian_single_stage
#SBATCH --partition=seasdean1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


THREADS="${SLURM_CPUS_PER_TASK:-24}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"

: "${FIELD_POLARITY:?Set FIELD_POLARITY to 1 or -1}"
: "${INIT_DIR:?Set INIT_DIR to the matching Stage-2 output directory}"
case "${FIELD_POLARITY}" in
  1|-1) ;;
  *) echo "FIELD_POLARITY must be 1 or -1" >&2; exit 2 ;;
esac

PYTHON_BIN=/burg-archive/home/tg2998/simsopt/venv/bin/python
srun --cpu-bind=cores "${PYTHON_BIN}" -u single_stage_dipoles.py \
      --init-dir "${INIT_DIR}" \
      --iota-target 0.064 \
      --f-cp-threshold 150000 \
      --field-polarity "${FIELD_POLARITY}" \
      --fb-threshold 5e-5 \
      --outer-step-radius 0.15
