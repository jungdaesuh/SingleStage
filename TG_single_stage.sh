#!/bin/bash -l
#SBATCH --account=seasdean        
#SBATCH --job-name=GoatedFAST   
#SBATCH --partition=seasdean1
#SBATCH -N 1              
#SBATCH -n 1                
#SBATCH -c 32                   
#SBATCH --mem=64G                
#SBATCH --time=10:00:00
#SBATCH --output=TG_single_stageFAST.out
#SBATCH --error=TG_single_stageFAST.err

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/burg-archive/home/tg2998/simsopt/venv/bin/python -u /burg-archive/home/tg2998/simsopt/examples/dipoles/TG_single_stage.py