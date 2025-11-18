#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J evo_train
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

DATASETS=(set5_exploit set5_explore set5_mixed set5_wide set5_mutated set5_sparse set5_dense)

nvidia-smi
for DATASET in "${DATASETS[@]}"; do
  echo "==== Training on ${DATASET} ===="
  python training.py \
    --dataset "${DATASET}" \
    --epochs 1000 \
    --optimizer LBFGS \
    --lr 1e-3 \
    --print-every 50
done

echo "Job finished at $(date)"