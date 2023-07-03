#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000

source /home4/s2614855/.venv/bin/activate

python /home4/s2614855/JWST/crop_images.py $*

deactivate