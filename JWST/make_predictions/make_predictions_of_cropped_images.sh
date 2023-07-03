#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=predictions_JWST

module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

source /home4/s2614855/.venv/bin/activate

python /home4/s2614855/JWST/make_predictions/make_predictions_of_cropped_images.py

deactivate