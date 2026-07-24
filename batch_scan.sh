#!/bin/bash
#SBATCH -A m4680
#SBATCH --job-name=single_stage
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128   # one process per core; each core runs one optimization
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

# Activate virtual environment
conda activate simsopt

# Number of runs: equal to the number of iota targets.
# Each run performs an internal continuation over current weights.
N=$(python3 generate_inputs.py --total)

# Stage 2 directory to initialize from
#INIT_DIR="../outputs/stage_2_npol11_ntor8_wout_nfp22ginsburg_000_000281/91_npol_11_ntor_8_VV_a_0.229_VV_b_0.265_VV_R0_1.038"
INIT_DIR="../outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_000281/90_npol_10_ntor_8_VV_a_0.250_VV_b_0.282_VV_R0_1.042"
#INIT_DIR="../outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_001242/08_npol_10_ntor_8_VV_a_0.245_VV_b_0.297_VV_R0_1.015"
#INIT_DIR="../outputs/stage_2_npol11_ntor8_wout_nfp22ginsburg_000_001242/01_npol_11_ntor_8_VV_a_0.235_VV_b_0.268_VV_R0_1.022"

VOL_TARGET=0.40

# How many runs to execute in parallel on this node and how many threads to use for each.
export OMP_NUM_THREADS=8        # or 8, 16 – you can experiment
MAX_JOBS=$(( SLURM_CPUS_PER_TASK / OMP_NUM_THREADS ))
echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS per job, MAX_JOBS=$MAX_JOBS"

running=0
completed=0

for i in $(seq 0 $((N-1))); do
    # Generate input parameters (iota target + continuation schedule)
    PARAMS=$(python3 generate_inputs.py --index $i)
    echo "[$((i+1))/$N] Starting optimization $i with parameters $PARAMS"

    # Launch this optimization in the background.
    python3 single_stage_epsilon_constraint.py --init-dir "$INIT_DIR" --vol-target $VOL_TARGET $PARAMS &
    ((running++))

    # If we've reached the concurrency limit, wait for at least one job to finish.
    if (( running >= MAX_JOBS )); then
        wait -n
        ((running--))
        ((completed++))
        echo "[$(date +%H:%M:%S)] One iteration finished — completed: $completed/$N, still running: $running"
    fi
done

# Wait for any remaining jobs to finish and print completion status.
while (( running > 0 )); do
    wait -n
    ((running--))
    ((completed++))
    echo "[$(date +%H:%M:%S)] One iteration finished — completed: $completed/$N, still running: $running"
done
echo "All $N iterations completed."