#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2                # number of GPUs per node
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --ntasks-per-node=1         # number of MP tasks

#SBATCH --nodes=1                  #Number of node. change this to >1 if you want to use more than one node

module load devel/cuda/12.9
source "$HOME"/nvflare/bin/activate
cd "$HOME"/basic_training_diffusion_mlflow_perun/

echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source .env
export GPUS_PER_NODE=2
echo "-----------------------------------------------------------"
nvidia-smi -l 3 >> nvidia-smi-${SLURM_JOB_ID}.log &
nvidia-smi --list-gpus

######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    "
 
export SCRIPT="basic_training_diffusion.py"

    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT " 
srun $CMD
 
