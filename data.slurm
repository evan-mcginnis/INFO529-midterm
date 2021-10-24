#!/bin/bash


#SBATCH --job-name=data
#SBATCH -e data.e.%A
#SBATCH -o data.o.%A
#SBATCH --partition=windfall
#SBATCH --ntasks=12
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=12 
#SBATCH --mail-user=evanmc@email.arizona.edu

# other common options. 
### means they are commented out
###SBATCH --account=PI
###SBATCH --partition=windfall
###SBATCH --gres=gpu:1
###SBATCH --mem-per-cpu=5gb
###SBATCH —-mem=1600gb
###SBATCH --mail-type=ALL

#conda init bash
#source ~/.bash_profile
#conda activate midterm
python makecsv.py -w "Dataset_Competition/Training/inputs_weather_train.npy" -o "Dataset_Competition/Training/inputs_others_train.npy" -g "Dataset_Competition/clusterID_genotype.npy" -y "Dataset_Competition/Training/yield_train.npy" -c train.csv
