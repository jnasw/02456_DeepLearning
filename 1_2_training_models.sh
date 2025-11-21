#!/bin/bash
#BSUB -q gpuv100
#BSUB -J evo_train
#BSUB -n 2
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Job started on $(hostname) at $(date)"
module swap cuda/12.1
source .venv/bin/activate
mkdir -p logs model/SM_AVR_GOV

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export WANDB_MODE=disabled

DATASETS=(set3_grid set5_exploit set5_explore set5_mixed set5_wide set5_mutated set5_sparse set5_dense)

nvidia-smi
for DATASET in "${DATASETS[@]}"; do
  echo "==== Training on ${DATASET} ===="
  python 1_2_training_models.py \
    --dataset "${DATASET}" \
    --epochs 1000 \
    --optimizer LBFGS \
    --lr 1e-3 \
    --print-every 50
done

echo "Job finished at $(date)"