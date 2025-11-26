#!/bin/bash
#BSUB -q gpuv100
#BSUB -J pinn_weighting
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

  

  # S0: Static weighting
  echo " Training with Static weighting "
  python 2_1_training_models.py \
      --dataset "${DATASET}" \
      --epochs 10000 \
      --weighting Static \
      --lr 1e-3 \
      --perc_data 1 \
      --perc_col 1 \
      --path "S0"

#  S1: MA weighting
  echo " Training with MA weighting "
  python 2_1_training_models.py \
      --dataset "${DATASET}" \
      --epochs 10000 \
      --weighting MA \
      --lr 1e-3 \
      --perc_data 1 \
      --perc_col 1 \
      --path "S1"

# S2a: Multiphase static
python 2_2_training_model.py \
    --dataset "${DATASET}" \
    --epochs 10000 \
    --lr 1e-3 \
    --perc_data 1 \
    --perc_col 1 \
    --switch_epoch 200 \
    --path "S2a"

# S2b: Multiphase MA
python 2_2_training_model.py \
    --dataset "${DATASET}" \
    --epochs 10000 \
    --weighting MA \
    --lr 1e-3 \
    --perc_data 1 \
    --perc_col 1 \
    --switch_epoch 200 \
    --path "S2b" 

# S3 Multi Stage static
python 2_3_training_model.py \
    --dataset "$DATASET" \
    --epochs 10000 \
    --perc_data 1.0 \
    --perc_col 1.0 \
    --switch_epoch 500 \
    --path "S3"

# S4 Multi Stage DN

python 2_4_training_model.py \
    --dataset ${DATASET} \
    --lbfgs_epochs 500 \
    --adam_epochs 10000 \
    --adam_lr 1e-3 \
    --perc_data 1.0 \
    --perc_col 1.0 \
    --update_freq 50 \
    --path "S4"

# S5 Multi Stage DN + warmup
python 2_4_training_model.py \
    --dataset ${DATASET} \
    --lbfgs_epochs 500 \
    --adam_epochs 10000 \
    --adam_lr 1e-3 \
    --perc_data 1.0 \
    --perc_col 1.0 \
    --update_freq 50 \
    --path "S5"

# S6 Multi Stage + warmup + MA
python 2_4_training_model.py \
    --dataset ${DATASET} \
    --lbfgs_epochs 500 \
    --adam_epochs 10000 \
    --adam_lr 1e-3 \
    --perc_data 1.0 \
    --perc_col 1.0 \
    --update_freq 50 \
    --path "S6"


echo "Job finished at $(date)"