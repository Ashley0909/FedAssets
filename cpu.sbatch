#!/bin/bash
#
#SBATCH --job-name=multicore-cpu # Job name for tracking
#SBATCH --partition=cpu-batch  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=40     # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --exclusive=mcs        # Exclusive mode, only this job will run
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=FAIL
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
python3 main.py
