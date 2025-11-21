#!/bin/bash
#BSUB -q gpuv100
#BSUB -J pinn_ma_static
#BSUB -n 2
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Job started on $(hostname) at $(date)"

module swap cuda/11.7
source .venv/bin/activate

mkdir -p logs model/SM_AVR_GOV

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export WANDB_MODE=disabled

DATASETS=(set5_mixed)

nvidia-smi

for DATASET in "${DATASETS[@]}"; do
  echo "========================================="
  echo "   DATASET: ${DATASET}"
  echo "========================================="

  # Model 1: MA weighting
  echo "---- Training with MA weighting ----"
  python 2_1_training_models.py \
      --dataset "${DATASET}" \
      --epochs 10000 \
      --weighting MA \
      --lr 1e-3 \
      --perc_data 1 \
      --perc_col 1

  # Model 2: Static weighting
  echo "---- Training with Static weighting ----"
  python 2_1_training_models.py \
      --dataset "${DATASET}" \
      --epochs 10000 \
      --weighting Static \
      --lr 1e-3 \
      --perc_data 1 \
      --perc_col 1

done

echo "Job finished at $(date)"