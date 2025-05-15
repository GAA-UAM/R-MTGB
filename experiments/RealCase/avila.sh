#!/bin/bash
#SBATCH --job-name=avila
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=fast
#SBATCH --output=avila.log

source /home/samanema/miniconda3/etc/profile.d/conda.sh
conda activate hfree

python run.py --dataset avila --seed 42
