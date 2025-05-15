#!/bin/bash

# Submit batch jobs using SLURM

sbatch -A ada2_serv -p cccmd school.sh
sbatch -A ada2_serv -p cccmd adult_gender.sh
sbatch -A ada2_serv -p cccmd adult_race.sh
sbatch -A ada2_serv -p cccmd parkinson.sh
sbatch -A ada2_serv -p cccmd computer.sh
sbatch -A ada2_serv -p cccmd landmine.sh
sbatch -A ada2_serv -p cccmd sarcos.sh
sbatch -A ada2_serv -p cccmd abalone.sh
sbatch -A ada2_serv -p cccmd avila.sh
sbatch -A ada2_serv -p cccmd bank.sh