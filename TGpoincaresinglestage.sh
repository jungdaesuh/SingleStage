#!/bin/bash -l
#SBATCH --account=seasdean     
#SBATCH --job-name=Poincare
#SBATCH --partition=seasdean1
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # 1 task (Since it's non-MPI serial Python)
#SBATCH -c 32                    # 1 CPU core for that task
#SBATCH --mem=128G                # 👇 64GB of RAM to feed them
#SBATCH --time=10:00:00
#SBATCH --output=TGpoincaresinglestage.out
#SBATCH --error=TGpoincaresinglestage.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

/burg-archive/home/tg2998/simsopt/venv/bin/python -u /burg-archive/home/tg2998/simsopt/examples/dipoles/TGpoincaresinglestage.py d1 d2