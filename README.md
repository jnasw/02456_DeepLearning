# PINNs for Power System Dynamics  
**02456 Deep Learning — Project Repository**

This repository contains the code and experiments for the 02456 Deep Learning project **evaluating sampling, loss-weighting, and adaptive collocation strategies** in Physics-Informed Neural Networks (PINNs).  
The study is conducted on the **9th-order SM_AVR_GOV synchronous machine model**, implemented using the PowerPINN framework.

The repository includes:
- Scripts for **dataset generation**
- Experiments for **sampling**, **weighting**, and **collocation**
- **HPC execution scripts** for reproducibility
- Jupyter notebooks for visualizing and reproducing the results

For a complete description of the methodology, experiments, and results, please refer to the **project report (PDF)** included at the top level of this repository.

## Repository Structure

A minimal overview of the repository layout is provided below.  
For detailed descriptions of all experiments and results, please refer to the project report.

```text
02456_DeepLearning/
├── README.md                      # Project README (this file)
├── 02456_PINNs_for_SM.pdf         # Project report (PDF)
├── requirements.txt               # Python dependencies
│
├── src/                           # PowerPINN framework code
│   ├── ode/                       # ODE models (e.g. SM_AVR_GOV)
│   ├── nn/                        # Neural network models & training logic
│   ├── dataset/                   # Dataset creation utilities
│   ├── conf/                      # YAML configuration files
│   └── ...
│
├── data/                          # Generated datasets
│   └── SM_AVR_GOV/
│
├── results/                       # Saved models and metrics
│   ├── 1_sampling/
│   ├── 2_weighting/
│   └── 3_collocation/
│
│── 1_sampling_analysis.ipynb      # Jupyter notebooks for analysis
│── 2_weighting_analysis.ipynb
│── 3_collocation_analysis.ipynb
├── 1_1_create_evo_datasets.py     # Sampling / dataset generation
├── 1_3_training_models.py         # Sampling experiments (local)
├── 1_2_training_models.sh         # Sampling experiments (HPC)
├── 2_0_training_models.sh         # Weighting experiments (HPC)
├── 2_1_training_models.py         # Weighting experiments (local)
├── 3_1_training_model.py          # Adaptive collocation experiment
│
└── LICENSE                        # Project license (MIT)
```

## PowerPINN Framework

This project builds on top of the **PowerPINN** framework, which provides utilities for
defining ODE-based power-system models, generating datasets, and training Physics-Informed
Neural Networks.  

PowerPINN handles:
- ODE definitions for the SM_AVR_GOV model  
- Dataset generation (trajectories + collocation points)  
- PINN architecture and training routines  
- YAML-based configuration of all experiments  

For full details on the framework design and equations, please refer to the original
PowerPINN paper and the project report included in this repository.

## Installation & Setup
Clone the repository and install the required Python packages:

```bash
git clone https://github.com/jnasw/02456_DeepLearning.git
cd 02456_DeepLearning
pip install -r requirements.txt
```

A GPU is recommended for model training but not required for dataset generation.

All experiment configurations (datasets, weighting schemes, collocation settings) are controlled through YAML files located in src/conf/ but manually modified in the individual scripts to enable reproducibility.

## Reproducibility & HPC Usage

All experiments in this project can be fully reproduced using the provided Python scripts and HPC job files. Each experiment (sampling, weighting, collocation) has its own training script(s) and HPC `.sh` file.

For HPC guides, refer to the corresponding website [here](https://www.hpc.dtu.dk).

Some notes for the recreation of the files investigated in the notebooks:
1) Sampling: The grid dataset using the frameworks create_init_cond_set3() function is already part of the data folder. To generate the investigated evo datasets, run the python script '1_1_create_evo_datasets.py' and the files will be stored in the data folder automatically. For the generated files of the model training, after running the hpc job please move the generated files under 'model/data' to the corresponding 'results/1_sampling' folder or use the already generated files.

2) Weighting: The sh file should generate all needed result csv files under the 'model/data_dt_pinn_ic' folder. In order to run the analysis notebook, please move the generated files to the 'results/2_weighting' folder. 

3) Collocation: Similarly, run the hpc script for the collocation results and move the generated results in the 'model/data_dt_pinn_ic' to the 'results/3_collocation' folder. For multiple runs, change the seed in the yaml file under 'src/conf/setup_dataset_nn.yaml'.
