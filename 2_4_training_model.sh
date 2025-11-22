#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pinn_multiphase_dn
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Job started on $(hostname) at $(date)"

module swap cuda/12.1
source .venv/bin/activate

mkdir -p logs model/SM_AVR_GOV

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export WANDB_MODE=disabled

DATASET="set5_mixed"

nvidia-smi

python 2_4_training_model.py \
    --dataset ${DATASET} \
    --lbfgs_epochs 500 \
    --adam_epochs 10000 \
    --adam_lr 1e-3 \
    --perc_data 1.0 \
    --perc_col 1.0 \
    --update_freq 50

echo "Job finished at $(date)"