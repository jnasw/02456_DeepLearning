#!/usr/bin/env python3
import os
import argparse
import torch
from omegaconf import OmegaConf

from src.nn.nn_dataset import DataSampler
from src.ode.sm_models_d import SynchronousMachineModels
from src.nn.nn_actions import NeuralNetworkActions


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
# ARGUMENT PARSER
# ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--perc_data", type=float, default=1.0)
    p.add_argument("--perc_col", type=float, default=1.0)
    p.add_argument("--adam_lr", type=float, default=1e-3)
    p.add_argument("--update_freq", type=int, default=5)

    # Epoch settings
    p.add_argument("--lbfgs_epochs", type=int, default=500)
    p.add_argument("--adam_epochs", type=int, default=10000)
    p.add_argument("--path", type=str, default=None)

    return p.parse_args()


# ----------------------------------------------
# MAIN TRAINING FUNCTION
# ----------------------------------------------
def main():
    args = parse_args()
    device = detect_device()

    
    # Load config YAML
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")

    # Full training horizon
    cfg.nn.num_epochs = args.lbfgs_epochs + args.adam_epochs
    cfg.nn.type = "DynamicNN"
    cfg.nn.early_stopping = False
    cfg.nn.path = args.path

    
    # MULTI-PHASE OPTIMIZER SETUP
    cfg.nn.multi_optim = True
    cfg.nn.switch_optim_epoch = args.lbfgs_epochs  # LBFGS → Adam

    # Phase 1: LBFGS
    cfg.nn.optimizer = "LBFGS"
    cfg.nn.lr = 1e-3          # LBFGS lr

    # Phase 2: Adam
    cfg.nn.optimizer_2.lr = args.adam_lr
    cfg.nn.optimizer_2.betas = [0.9, 0.999]

    
    # DN WEIGHTING CONFIG  
    cfg.nn.weighting.update_weight_method = "DN"
    cfg.nn.weighting.update_weights_freq = args.update_freq
    cfg.nn.weighting.Beta = 0.99
    cfg.nn.weighting.bias_correction = True

    # *** IMPORTANT ***
    # Phase 1 data-only but unfrozen (mask stays all 1's)
    cfg.nn.weighting.weights = [1.0, 1e-9, 1e-9, 1e-9]

    
    # Dataset sampling
    cfg.dataset.perc_of_data_points = args.perc_data
    cfg.dataset.perc_of_col_points = args.perc_col

    
    # Dataset loading
    path = f"data/SM_AVR_GOV/dataset_{args.dataset}.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f" Dataset: {path}")
    print(f" LBFGS epochs: {args.lbfgs_epochs}")
    print(f" Adam epochs:  {args.adam_epochs}")
    print(f" Switch at:    epoch {cfg.nn.switch_optim_epoch}")

    ds = DataSampler(cfg, dataset_path=path)
    modelling = SynchronousMachineModels(cfg)

    
    # Initialize NN trainer
    net = NeuralNetworkActions(cfg, modelling, data_loader=ds)
    net.model.to(device)

    # RUN TRAINING (LBFGS → Adam transitions automatically)
    net.pinn_train2(
        num_of_skip_data_points=1,
        num_of_skip_col_points=1,
        num_of_skip_val_points=1,
        wandb_run=None,
    )

    # SUMMARY
    print("Training completed.")
    print(f"Final total loss: {net.loss_total.item():.4e}")
    print(f"Data loss:        {net.loss_data.item():.4e}")
    print(f"DT loss:          {net.loss_dt.item():.4e}")
    print(f"PINN loss:        {net.loss_pinn.item():.4e}")
    print(f"IC loss:          {net.loss_pinn_ic.item():.4e}")


if __name__ == "__main__":
    main()