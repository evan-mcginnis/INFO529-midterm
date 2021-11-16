#!/bin/bash


#SBATCH --job-name=train-singularity
#SBATCH -e train.e.%A
#SBATCH -o train.o.%A
#SBATCH --partition=windfall
#SBATCH --ntasks=4
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12 
#SBATCH --mem-per-cpu=24gb

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

singularity run -B $HOME/INFO529-midterm/data:/data info529_latest.sif

