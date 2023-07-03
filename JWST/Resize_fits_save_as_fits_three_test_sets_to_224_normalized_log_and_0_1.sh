#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80GB

source /home4/s2614855/.venv/bin/activate

python /home4/s2614855/JWST/Resize_fits_save_as_fits_three_test_sets_to_224_normalized_log_and_0_1.py $*

deactivate