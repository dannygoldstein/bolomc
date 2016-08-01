#!/bin/bash

#SBATCH -p regular
#SBATCH -t 12:00:00
#SBATCH -N 24
#SBATCH -J JOBNAME
#SBATCH -L SCRATCH
#SBATCH --mail-type=ALL
#SBATCH -A m1400
#SBATCH -e JOBNAME.e
#SBATCH -o JOBNAME.o

srun -n 24 python fit.py xx.yml
