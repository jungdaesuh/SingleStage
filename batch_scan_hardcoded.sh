#!/bin/bash
#SBATCH -A m4680
#SBATCH --job-name=single_stage
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32   # one process per core; each core runs one optimization
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com

# Purge all modules to remove the "Application linked against multiple cray-libsci libraries" warning
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh

# Load Python module
module load python/3.11

# Activate virtual environment
conda activate simsopt

# Redirect stdout/stderr to a dated, run-count, job-specific log file:
#   YYMMDD_<N>_<JOBID>.out
OUT_DIR="../single_stage_scans_no_sparsity/slurm_outputs"
mkdir -p "$OUT_DIR"
DATE_STR=$(date +%y%m%d)
OUT_FILE="${OUT_DIR}/${DATE_STR}_${N}_${SLURM_JOB_ID:-local}.out"
exec >"$OUT_FILE" 2>&1

# How many runs to execute in parallel on this node and how many threads to use for each.
export OMP_NUM_THREADS=32
# echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS per job, MAX_JOBS=8"

# python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/79_20260312_diprad_0.051_VV_a_0.247_VV_b_0.274_VV_R0_1.019' --iota-target=0.0669462 --iota-weight=72.9079 --qs-weight=166.65 --current-threshold=194089.0 --current-weight=3.18385 &

# python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/79_20260312_diprad_0.051_VV_a_0.247_VV_b_0.274_VV_R0_1.019' --iota-target=0.0647183 --iota-weight=200.68 --qs-weight=1.20727 --current-threshold=157524.0 --current-weight=5.36227 &

# python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/59_20260312_diprad_0.050_VV_a_0.263_VV_b_0.282_VV_R0_1.028' --iota-target=0.0663147 --iota-weight=396.062 --qs-weight=8.47198 --current-threshold=163625.0 --current-weight=3.00987 &

# python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/59_20260312_diprad_0.050_VV_a_0.263_VV_b_0.282_VV_R0_1.028' --iota-target=0.0772105 --iota-weight=170.374 --qs-weight=92.9325 --current-threshold=163999.0 --current-weight=3.53249 &   

# python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/79_20260312_diprad_0.051_VV_a_0.247_VV_b_0.274_VV_R0_1.019' --iota-target=0.0774713 --iota-weight=190.831 --qs-weight=1.19336 --current-threshold=189846.0 --current-weight=2.35824 &

# python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/79_20260312_diprad_0.051_VV_a_0.247_VV_b_0.274_VV_R0_1.019' --iota-target=0.0809006 --iota-weight=139.282 --qs-weight=1.30226 --current-threshold=194196.0 --current-weight=2.14564 &

# python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/79_20260312_diprad_0.051_VV_a_0.247_VV_b_0.274_VV_R0_1.019' --iota-target=0.0885271 --iota-weight=97.7176 --qs-weight=171.36 --current-threshold=170071.0 --current-weight=3.60225 &

python3 single_stage_dipole_example.py --init-dir='/global/u1/j/jhalpern/codes/simsopt/examples/outputs/stage_2_LHS_3D_scan_no_sparsity/59_20260312_diprad_0.050_VV_a_0.263_VV_b_0.282_VV_R0_1.028' --iota-target=0.115549 --iota-weight=191.563 --qs-weight=305.691 --current-threshold=151233.0 --current-weight=2.14109 #&

wait
echo "All runs complete."
