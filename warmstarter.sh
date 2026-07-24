#!/bin/bash -l
#SBATCH --account=apam        
#SBATCH --job-name=warmU
#SBATCH --partition=apam1
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # 1 task (Since it's non-MPI serial Python)
#SBATCH -c 8                     # 1 CPU core for that task
#SBATCH --mem=4G               
#SBATCH --time=3:00:00
#SBATCH --output=warmstarterU.out
#SBATCH --error=warmstarterU.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

/burg-archive/home/tg2998/simsopt/venv/bin/python -u warmstarter.py 