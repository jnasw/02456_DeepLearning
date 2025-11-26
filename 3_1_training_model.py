#!/usr/bin/env python3
import os
import argparse
import torch
from omegaconf import OmegaConf

from src.nn.nn_dataset import DataSampler
from src.ode.sm_models_d import SynchronousMachineModels
from src.nn.nn_actions import NeuralNetworkActions


# DEVICE SELECTION
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ARG PARSER (LBFGS-only)
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.addargument("--num_epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)

    # dataset size
    p.add_argument("--perc_data", type=float, default=1.0)
    p.add_argument("--perc_col", type=float, default=1.0)

    # adaptive collocation
    p.add_argument("--adaptive_enabled", action="store_true")
    p.add_argument("--adapt_every", type=int, default=50)
    p.add_argument("--adapt_ratio", type=float, default=0.3)
    p.add_argument("--adapt_strategy", type=str, default="resample")
    p.add_argument("--save_snapshots", action="store_true")

    # eval grid
    p.add_argument("--eval_n_time", type=int, default=50)
    p.add_argument("--eval_theta_min", type=float, default=-2.0)
    p.add_argument("--eval_theta_max", type=float, default= 2.0)
    p.add_argument("--eval_omega_min", type=float, default=-1.0)
    p.add_argument("--eval_omega_max", type=float, default= 1.0)

    return p.parse_args()


# MAIN
def main():
    args = parse_args()
    device = detect_device()

    # Load YAML config
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")

    # BASE TRAINING CONFIG
    cfg.nn.type = "DynamicNN"
    cfg.nn.num_epochs = args.num_epochs
    cfg.nn.lr = args.lr
    cfg.nn.early_stopping = False

    # LBFGS ONLY
    cfg.nn.multi_optim = False
    #cfg.nn.optimizer = "LBFGS"

    # NON-BATCH MODE REQUIRED FOR ADAPTIVE COLLOCATION
    cfg.nn.batch_size = "None"

    # static weighting only
    cfg.nn.weighting.update_weight_method = "Static"
    #cfg.nn.weighting.weights = [1.0, 1e-3, 1e-3, 1e-3]
    cfg.nn.weighting.update_weights_freq = 999999

    # dataset subsets
    cfg.dataset.perc_of_data_points = args.perc_data
    cfg.dataset.perc_of_col_points = args.perc_col

    # Adaptive collocation config
    cfg.nn.adaptive_col = OmegaConf.create({
        "enabled": args.adaptive_enabled,
        "adapt_every": args.adapt_every,
        "ratio": args.adapt_ratio,
        "strategy": args.adapt_strategy,
        "save_snapshots": args.save_snapshots,
        "eval_grid": {
            "n_time": args.eval_n_time,
            "theta_range": [args.eval_theta_min, args.eval_theta_max],
            "omega_range": [args.eval_omega_min, args.eval_omega_max],
        }
    })

    # load dataset
    dataset_path = f"data/SM_AVR_GOV/dataset_{args.dataset}.pkl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ds = DataSampler(cfg, dataset_path=dataset_path)
    modelling = SynchronousMachineModels(cfg)

    # trainer
    net = NeuralNetworkActions(cfg, modelling, data_loader=ds)
    net.model.to(device)

    # TRAIN (LBFGS-only)
    net.pinn_train2(
        num_of_skip_data_points=1,
        num_of_skip_col_points=1,
        num_of_skip_val_points=1,
        wandb_run=None,
    )

    # summary
    print("\nLBFGS TRAINING COMPLETE ")
    print(f"Final total loss: {net.loss_total.item():.4e}")
    print(f"Data loss       : {net.loss_data.item():.4e}")
    print(f"dt loss         : {net.loss_dt.item():.4e}")
    print(f"PINN loss       : {net.loss_pinn.item():.4e}")
    print(f"IC loss         : {net.loss_pinn_ic.item():.4e}")


if __name__ == "__main__":
    main()