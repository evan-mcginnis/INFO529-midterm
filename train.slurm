#!/bin/bash


#SBATCH --job-name=train
#SBATCH -e train.e%A
#SBATCH -o train.o%A
#SBATCH --partition=windfall
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=12 

# other common options. 
### means they are commented out
###SBATCH --account=PI
###SBATCH --partition=windfall
###SBATCH --gres=gpu:1
###SBATCH --mem-per-cpu=5gb
###SBATCH —-mem=1600gb
###SBATCH --mail-type=ALL
###SBATCH --mail-user=netid@email.arizona.edu

# Warning: This script does not yet work
#conda init bash
#conda activate midterm
python predict-yield.py -m model_midterm.ckpt -i 350000 -d train.npz

