#!/bin/bash

#PBS -q debug
#PBS -l mppwidth=24
#PBS -l walltime=00:30:00
#PBS -N MAKE_BOLOMETRIC

module load python/2.7-anaconda
aprun -n 24 python-mpi fit.py xx.yml



