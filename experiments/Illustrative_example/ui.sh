#!/bin/bash
#SBATCH --job-name=ui
#SBATCH --nodes=1                   
#SBATCH --ntasks=3
#SBATCH --time=120:00:00             
#SBATCH --output=ui.log


source /home/samanema/miniconda3/etc/profile.d/conda.sh
conda activate hfree

python ui_exp.py
