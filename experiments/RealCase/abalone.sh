#!/bin/bash
#SBATCH --job-name=abalone
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=fast
#SBATCH --output=abalone.log

source /home/samanema/miniconda3/etc/profile.d/conda.sh
conda activate hfree

python run.py --dataset abalone --seed 42


