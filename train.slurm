#!/bin/bash


#SBATCH --job-name=train4
#SBATCH -e train.e.%A
#SBATCH -o train.o.%A
#SBATCH --partition=windfall
#SBATCH --ntasks=4
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12 
#SBATCH --mem-per-cpu=24gb
#SBATCH --mail-user=evanmc@email.arizona.edu

# other common options. 
### means they are commented out
###SBATCH --account=PI
###SBATCH --partition=windfall
###SBATCH --gres=gpu:1
###SBATCH --mem-per-cpu=5gb
###SBATCH —-mem=1600gb
###SBATCH --mem=1600gb
###SBATCH --mail-type=ALL
###SBATCH --mail-user=netid@email.arizona.edu

conda init bash
source ~/.bash_profile
conda activate midterm
python predict-yield.py -m model_midterm.ckpt -i 350000 -d train.npz

