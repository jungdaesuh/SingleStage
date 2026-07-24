#!/bin/bash
#SBATCH -A m4680
#SBATCH --job-name=ss_rerun
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

module purge 2>/dev/null || true
if [[ -f /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh ]]; then
  source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
fi
module load python/3.11 2>/dev/null || true
if command -v conda &>/dev/null; then
  conda activate simsopt 2>/dev/null || true
fi

export OMP_NUM_THREADS=128

INIT_91="../outputs/stage_2_npol11_ntor8_wout_nfp22ginsburg_000_000281/91_npol_11_ntor_8_VV_a_0.229_VV_b_0.265_VV_R0_1.038"

echo "========== init_dir91, iota_target=0.23 =========="
python3 single_stage_epsilon_constraint.py \
  --init-dir "$INIT_91" \
  --iota-target 0.23 \
  --iota-weight 1.0 \
  --qs-weight 1.0 \
  --current-weight-schedule 0.25,0.5,1.0,2.0

echo "Finished."
