#!/usr/bin/env python3
import os
import argparse
from omegaconf import OmegaConf
import torch

from src.nn.nn_dataset import DataSampler
from src.nn.nn_actions import NeuralNetworkActions
from src.ode.sm_models_d import SynchronousMachineModels


def detect_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")
    return dev


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--weighting", type=str,
                   choices=["MA", "Static"], default="MA")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--perc_data", type=float, default=1.0)
    p.add_argument("--perc_col", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()
    device = detect_device()

    # Load configuration
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")
    cfg.nn.type = "DynamicNN"
    cfg.nn.num_epochs = args.epochs
    cfg.nn.early_stopping = False
    cfg.nn.weighting.update_weight_method = args.weighting
    cfg.dataset.perc_of_data_points = args.perc_data
    cfg.dataset.perc_of_col_points = args.perc_col

    if args.lr is not None:
        cfg.nn.lr = args.lr

    # Dataset path
    dataset_path = f"data/SM_AVR_GOV/dataset_{args.dataset}.pkl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"\n=== Training on dataset: {dataset_path}")
    print(f"=== Weighting: {args.weighting}")
    print(f"=== Epochs: {args.epochs}")
    print(f"=== Using device: {device}\n")

    # Load data + physics model
    ds = DataSampler(cfg, dataset_path=dataset_path)
    modelling_full = SynchronousMachineModels(cfg)

    # Create PINN trainer
    net = NeuralNetworkActions(cfg, modelling_full, data_loader=ds)
    net.model.to(device)

    # Run the actual training loop (LBFGS inside)
    net.pinn_train2(
        num_of_skip_data_points=1,
        num_of_skip_col_points=1,
        num_of_skip_val_points=1,
        wandb_run=None,
    )

    # Summary
    print("\nTraining completed.")
    print(f"Final total loss: {net.loss_total.item():.4e}")
    print(f"Data loss:        {net.loss_data.item():.4e}")
    print(f"dt loss:          {net.loss_dt.item():.4e}")
    print(f"PINN loss:        {net.loss_pinn.item():.4e}")
    print(f"IC loss:          {net.loss_pinn_ic.item():.4e}")


if __name__ == "__main__":
    main()