#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pinn_ac_lbfgs
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

python train_ac_lbfgs.py \
    --dataset ${DATASET} \
    --num_epochs 2000 \
    --lr 1e-3 \
    --perc_data 1.0 \
    --perc_col 1.0 \
    --adaptive_enabled \
    --adapt_every 50 \
    --adapt_ratio 0.3 \
    --adapt_strategy resample \
    --save_snapshots \
    --eval_n_time 50 \
    --eval_theta_min -2.0 \
    --eval_theta_max 2.0 \
    --eval_omega_min -1.0 \
    --eval_omega_max 1.0

echo "Job finished at $(date)"