#!/bin/bash
#SBATCH --job-name=bank
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=fast
#SBATCH --output=bank.log

source /home/samanema/miniconda3/etc/profile.d/conda.sh
conda activate hfree

python run.py --dataset bank --seed 42
