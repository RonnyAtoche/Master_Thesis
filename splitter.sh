#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
 
module purge
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
 
source /data/s2614855/.envs/intro_project_venv/bin/activate

python splitter.py

deactivate