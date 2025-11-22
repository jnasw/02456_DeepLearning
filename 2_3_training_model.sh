#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pinn_lbfgs_adam_test
#BSUB -n 8
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
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

python train_multiphase.py \
    --dataset "$DATASET" \
    --epochs 10500 \
    --perc_data 1.0 \
    --perc_col 1.0 \
    --switch_epoch 500

echo "Job finished at $(date)"