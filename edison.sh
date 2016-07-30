#!/bin/bash

#SBATCH -p debug 
#SBATCH -t 00:30:00
#SBATCH -N 24
#SBATCH -J MAKE_BOLOMETRIC
#SBATCH -L scratch1

module load python/2.7-anaconda
srun -n 24 python-mpi fit.py xx.yml



