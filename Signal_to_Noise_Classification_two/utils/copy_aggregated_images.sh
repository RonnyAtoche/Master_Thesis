#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=80GB
 
module purge
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
 
source /data/s2614855/.envs/intro_project_venv/bin/activate

python /data/s2614855/Signal_to_Noise_Classification_two/utils/copy_aggreagted_images.py

deactivate