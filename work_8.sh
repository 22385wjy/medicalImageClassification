#!/bin/bash
#SBATCH -J wtest
#SBATCH -N 1 -n 8
#SBATCH --gpus=1
#SBATCH -o job8_%j.out
#SBATCH -e job8_%j.err
source activate pytorch38
echo This job runs on the following nodes:${SLURM_JOB_NODELIST}
echo start on $(date)
python 8ADNCEMCILMCI_4classify_test.py 
echo end on $(date)
