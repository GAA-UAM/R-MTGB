#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --nodes=1                   
#SBATCH --ntasks=3
#SBATCH --time=120:00:00             
#SBATCH --output=gen_data.log


source /home/samanema/miniconda3/etc/profile.d/conda.sh
conda activate hfree

python gen_data.py
