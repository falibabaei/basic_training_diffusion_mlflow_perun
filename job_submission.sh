#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:0


source "$HOME"/.bashrc
echo 'Starting script'
source .env  

source my_env/bin/activate

python basic_training_diffusion.py
