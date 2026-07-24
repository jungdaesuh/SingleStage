#!/bin/bash -l
#SBATCH --account=apam   
#SBATCH --job-name=Current
#SBATCH --partition=apam1
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # 1 task (Since it's non-MPI serial Python)
#SBATCH -c 4                    # 1 CPU core for that task
#SBATCH --mem=1G                # 👇 64GB of RAM to feed them
#SBATCH --time=1:00:00
#SBATCH --output=TGPlotsCurrent.out
#SBATCH --error=TGPlotsCurrent.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/burg-archive/home/tg2998/simsopt/venv/bin/python -u /burg-archive/home/tg2998/simsopt/examples/dipoles/TGPlotsCurrent.py