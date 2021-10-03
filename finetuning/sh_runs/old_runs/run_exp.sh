#!/bin/sh
#SBATCH --time=2-0:0:0
#SBATCH --gres=gpu:rtx2080:1
source  /cs/labs/gabis/ednussi/v2/bin/activate
python augment_utils.py


