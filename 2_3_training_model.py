#!/usr/bin/env python3
import os
import argparse
from omegaconf import OmegaConf
import torch

from src.nn.nn_dataset import DataSampler
from src.nn.nn_actions import NeuralNetworkActions
from src.ode.sm_models_d import SynchronousMachineModels


# ----------------------------------------------
# DEVICE SELECTION
# ----------------------------------------------
def detect_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")
    return dev


# ----------------------------------------------
# ARGUMENTS
# ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--perc_data", type=float, default=1.0)
    p.add_argument("--perc_col", type=float, default=1.0)
    p.add_argument("--switch_epoch", type=int, default=200)
    p.add_argument("--path", type=str, default=None)

    return p.parse_args()

# MAIN TRAINING FUNCTION
def main():
    args = parse_args()
    device = detect_device()

     # Load config
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")

    cfg.nn.type = "DynamicNN"
    cfg.nn.early_stopping = False
    cfg.nn.path = args.path

    # HPC TWO-PHASE OPTIMIZER SETUP
    cfg.nn.num_epochs = 10000              # 500 LBFGS + 10000 Adam
    cfg.nn.multi_optim = True
    cfg.nn.switch_optim_epoch = 500        # switch LBFGS → Adam

    # Phase 1 weights (data only)
    cfg.nn.weighting.update_weight_method = "Static"
    cfg.nn.weighting.weights = [1.0, 0.0, 0.0, 0.0]

    # Phase 2 optimizer parameters (Adam)
    cfg.nn.optimizer_2.lr = 1e-3
    cfg.nn.optimizer_2.betas = [0.9, 0.999]

    # Dataset sampling
    cfg.dataset.perc_of_data_points = args.perc_data
    cfg.dataset.perc_of_col_points = args.perc_col

    print("\n=== Multi-optimizer Training ===")
    print(f"LBFGS → Adam at epoch {cfg.nn.switch_optim_epoch}")
    print("==============================================\n")

    # Dataset loading
    dataset_path = f"data/SM_AVR_GOV/dataset_{args.dataset}.pkl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ds = DataSampler(cfg, dataset_path=dataset_path)
    modelling_full = SynchronousMachineModels(cfg)

    # Initialize PINN trainer
    net = NeuralNetworkActions(cfg, modelling_full, data_loader=ds)
    net.model.to(device)

    # RUN TRAINING
    net.pinn_train2(
        num_of_skip_data_points=1,
        num_of_skip_col_points=1,
        num_of_skip_val_points=1,
        wandb_run=None,
    )

    # PRINT SUMMARY
    print("Training completed.")
    print(f"Final total loss: {net.loss_total.item():.4e}")
    print(f"Data loss:        {net.loss_data.item():.4e}")
    print(f"dt loss:          {net.loss_dt.item():.4e}")
    print(f"PINN loss:        {net.loss_pinn.item():.4e}")
    print(f"IC loss:          {net.loss_pinn_ic.item():.4e}")