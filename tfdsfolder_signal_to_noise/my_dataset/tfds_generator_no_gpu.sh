#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80GB
 
module purge
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
 
source /data/s2614855/.envs/intro_project_venv/bin/activate

cd /data/s2614855/Intro_Project_Data/Intro_Project/tfdsfolder/my_dataset/ && tfds build --overwrite

deactivate