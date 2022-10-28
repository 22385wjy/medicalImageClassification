#!/bin/bash
#SBATCH -J wtest
#SBATCH -N 1 -n 8
#SBATCH --gpus=1
#SBATCH -o job0_%j.out
#SBATCH -e job0_%j.err
source activate pytorch38
echo This job runs on the following nodes:${SLURM_JOB_NODELIST}
echo start on $(date)
python 0ADNC_2classify_test.py 
echo end on $(date)
