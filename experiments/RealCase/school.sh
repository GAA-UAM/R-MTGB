#!/bin/bash
#SBATCH --job-name=school
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=fast
#SBATCH --output=school.log

source /home/samanema/miniconda3/etc/profile.d/conda.sh
conda activate hfree

python school.py

