#!/bin/bash -l
#SBATCH --account=seasdean        
#SBATCH --job-name=0.300
#SBATCH --partition=seasdean1
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # 1 task (Since it's non-MPI serial Python)
#SBATCH -c 32                     # 1 CPU core for that task
#SBATCH --mem=32G                # 👇 64GB of RAM to feed them
#SBATCH --time=10:00:00
#SBATCH --output=single_stage_dipolesplsbroken1.out
#SBATCH --error=single_stage_dipolesplsbroken1.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

/burg-archive/home/tg2998/simsopt/venv/bin/python -u single_stage_dipoles.py \
      --init-dir /burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results/TF_a_0.300 \
      --iota-target 0.064 \
      --f-cp-threshold 150000 \
      --resolutions 6 \
      --fb-thresholds 1e-4