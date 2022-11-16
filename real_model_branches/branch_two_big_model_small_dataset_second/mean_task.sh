#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40:2
#SBATCH --mem=125GB
 
module purge
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4

source /data/s2614855/.envs/intro_project_venv/bin/activate

python /data/s2614855/Intro_Project_Data/Intro_Project/real_model_branches/branch_two_big_model_small_dataset_second/mean_task.py

deactivate