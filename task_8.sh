#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=task_8

module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

source /home4/s2614855/.venv/bin/activate

python /home4/s2614855/experiment_5/model_tasks/main_tasks/task_8.py

deactivate