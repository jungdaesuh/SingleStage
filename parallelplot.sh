#!/bin/bash -l
#SBATCH --account=apam        
#SBATCH --job-name=PlotParallel2   
#SBATCH --partition=apam1
#SBATCH -N 1              
#SBATCH -n 1                
#SBATCH -c 12                   
#SBATCH --mem=150G                
#SBATCH --time=10:00:00
#SBATCH --output=parallelplot2.out
#SBATCH --error=parallelplot2.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi

# Prevents thread oversubscription during Python multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run the parallel script using your virtual environment and unbuffered output
/burg-archive/home/tg2998/simsopt/venv/bin/python -u /burg-archive/home/tg2998/simsopt/examples/dipoles/parallelplot.py