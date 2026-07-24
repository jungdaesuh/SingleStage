#!/bin/bash -l
#SBATCH --account=apam        
#SBATCH --job-name=Tiki2
#SBATCH --partition=apam1
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # 1 task (Since it's non-MPI serial Python)
#SBATCH -c 24                     # 1 CPU core for that task
#SBATCH --mem=64G                # 👇 64GB of RAM to feed them
#SBATCH --time=48:00:00
#SBATCH --output=TG2PlotsStageII.out
#SBATCH --error=TG2PlotsStageII.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

/burg-archive/home/tg2998/simsopt/venv/bin/python -u /burg-archive/home/tg2998/simsopt/examples/dipoles/TGPlotsStageII.py