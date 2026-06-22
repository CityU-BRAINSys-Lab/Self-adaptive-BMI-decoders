# Self-adaptive-BMI-decoders

[![Paper](https://img.shields.io/badge/Paper-IOPscience-blue)](https://iopscience.iop.org/article/10.1088/2634-4386/ae6728/meta)
[![arXiv](https://img.shields.io/badge/arXiv-2511.22108-B31B1B.svg)](https://arxiv.org/abs/2511.22108)
[![Language](https://img.shields.io/badge/Language-Python-3776AB.svg)](https://www.python.org/)

This repository contains the official code implementation for the paper: 
**"An energy-efficient spiking neural network with continuous learning for self-adaptive brain-machine interface"** published at the **City University of Hong Kong (CityU)**.

---

## 📌 Overview

The number of simultaneously recorded neurons follows an exponentially increasing trend in implantable brain–machine interfaces (iBMIs). Integrating the neural decoder in the implant is an effective data compression method for future wireless iBMIs. However, the non-stationarity of the system makes the performance of the decoder unreliable. To avoid frequent retraining of the decoder and to ensure the safety and comfort of the iBMI user, continuous learning is essential for real-life applications. 

### Key Features
* **High Energy Efficiency:** There is 98% fewer computations compared to conventional continuous learning SNN decoders, matching wireless iBMI resource constraints.
* **Continuous Adaptive Learning:** Effectively mitigates the performance degradation caused by non-stationary neural signals.
* **Comprehensive Validation:** Includes code for both **Open-loop** simulation analysis and **Closed-loop** experimental validation.

---

## 📂 Repository Structure

```text
├── Open_loop/              # Implementation & scripts for open-loop decoding tasks
├── Closed_loop/            # Implementation & scripts for closed-loop self-adaptive tasks
├── datasets/               # Directory placeholder or scripts for neural datasets
├── model_weights/          # Pre-trained model weights for the SNN decoders
├── utils_files/            # Utility and helper functions (closed-loop environments, utility functions, etc.)
├── Figure_plot.py          # Script to reproduce the figures in the paper
├── energy_model.py         # The energy estimation models for the simulation
├── primate_reaching.py     # The preprocessing scripts of UCSF dataset
├── t_test.py               # Statistical evaluation and significance test scripts
├── closed_loop_figures.vsdx# Source visualization diagram for the closed-loop paradigm
└── requirements.txt        # Required packages and software environment list

## Installation & Prerequisites

# Clone the repository
git clone [https://github.com/CityU-BRAINSys-Lab/Self-adaptive-BMI-decoders.git](https://github.com/CityU-BRAINSys-Lab/Self-adaptive-BMI-decoders.git)
cd Self-adaptive-BMI-decoders

# Install dependencies
pip install -r requirements.txt

## How to Run

1. Open-loop Decoding
To train or evaluate the DSNN decoder on standard pre-recorded neural datasets:
cd Open_loop
python UCSF_open_loop.py  # Training
python Open_loop_inference.py # Inference

2. Closed-loop Decoding
To simulate the self-adaptive continuous learning process (using Banditron/AGREL updates) for closed-loop interactions:
cd Closed_loop
python OPS_closed_loop_electrode_shift.py  # For electrode shift experiment
python OPS_closed_loop_firing_rate_drift.py  # For firing rate drift experiment
python OPS_closed_loop_loss_neuron.py  # For loss of neurons experiment


## Citation
If you find this work, code, or decoders useful for your research, please consider citing our paper:

@article{biyan2026energy,
  title={An energy-efficient spiking neural network with continuous learning for self-adaptive brain--machine interface},
  author={Biyan, Zhou and Basu, Arindam},
  journal={Neuromorphic Computing and Engineering},
  volume={6},
  number={2},
  pages={024010},
  year={2026},
  publisher={IOP Publishing}
}






