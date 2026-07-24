#!/bin/bash -l
#SBATCH --account=seasdean    
#SBATCH --job-name=extraction
#SBATCH --partition=seasdean1
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # 1 task (Since it's non-MPI serial Python)
#SBATCH -c 4                     # 1 CPU core for that task
#SBATCH --mem=8G               
#SBATCH --time=0:10:00
#SBATCH --output=TGextraction.out
#SBATCH --error=TGextraction.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/burg-archive/home/tg2998/simsopt/venv/bin/python -u TGextraction.py