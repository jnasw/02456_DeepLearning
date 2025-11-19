#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pinn_ma_static
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
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
      --epochs 1000 \
      --weighting MA \
      --lr 1e-3 \
      --perc_data 1 \
      --perc_col 1

  # Model 2: Static weighting
  echo "---- Training with Static weighting ----"
  python 2_1_training_models.py \
      --dataset "${DATASET}" \
      --epochs 1000 \
      --weighting Static \
      --lr 1e-3 \
      --perc_data 1 \
      --perc_col 1

done

echo "Job finished at $(date)"