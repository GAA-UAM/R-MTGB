#!/bin/bash
#SBATCH --job-name=parkinson
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=fast
#SBATCH --output=parkinson.log

source /home/samanema/miniconda3/etc/profile.d/conda.sh
conda activate hfree



RECEIVED_DATASET="parkinson"
RECEIVED_SEED="0"

echo "-------------------------------------"
echo "Worker: Script started."
echo "Worker: Received Dataset: ${RECEIVED_DATASET}"
echo "Worker: Received seed: ${RECEIVED_SEED}"
echo "-------------------------------------"

echo "Running Python script..."

python run.py --dataset ${RECEIVED_DATASET} --seed ${RECEIVED_SEED}

sleep 2
echo "Python script finished."
exit 0
