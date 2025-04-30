#!/bin/bash

sbatch -A ada2_serv -p cccmd adult_gender.sh
sbatch -A ada2_serv -p cccmd adult_race.sh
sbatch -A ada2_serv -p cccmd landmine.sh
sbatch -A ada2_serv -p cccmd parkinson.sh
sbatch -A ada2_serv -p cccmd computer.sh
sbatch -A ada2_serv -p cccmd school.sh
