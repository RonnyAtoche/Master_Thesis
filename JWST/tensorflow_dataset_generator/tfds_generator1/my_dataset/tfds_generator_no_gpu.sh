#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=10
#SBATCH --mem=85GB

source /home4/s2614855/.venv/bin/activate

cd /home4/s2614855/JWST/tensorflow_dataset_generator/tfds_generator1/my_dataset/ && tfds build --overwrite

deactivate