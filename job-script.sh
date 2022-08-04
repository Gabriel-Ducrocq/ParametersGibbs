#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=90
#SBATCH --array=0-0

srun python main.py
