#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pinn_two_phase
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Job started on $(hostname) at $(date)"

# -------------------------------------------------------------
# ENVIRONMENT SETUP
# -------------------------------------------------------------
module swap cuda/12.1
source .venv/bin/activate

mkdir -p logs model/SM_AVR_GOV

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export WANDB_MODE=disabled

# -------------------------------------------------------------
# DATASET SELECTION
# -------------------------------------------------------------
DATASET="set5_mixed"    # CHANGE IF NEEDED

nvidia-smi

echo "========================================="
echo "   TRAINING TWO-PHASE MODEL"
echo "   DATASET: ${DATASET}"
echo "========================================="

# -------------------------------------------------------------
# TWO-PHASE TRAINING
#   Phase 1: data-only (weights [1,0,0,0])
#   Phase 2: static weights computed at switch_epoch
# -------------------------------------------------------------

python train_two_phase.py \
    --dataset "${DATASET}" \
    --epochs 15000 \
    --lr 1e-3 \
    --perc_data 1 \
    --perc_col 1 \
    --switch_epoch 200

echo "========================================="
echo "Training finished."
echo "Job ended at $(date)"
echo "========================================="