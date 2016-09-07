#!/bin/bash

#SBATCH -p regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -J JOBNAME
#SBATCH -L SCRATCH
#SBATCH --mail-type=ALL
#SBATCH -A m1186
#SBATCH -e JOBNAME.e
#SBATCH -o JOBNAME.o

srun -n 24 python fit.py xx.yml
