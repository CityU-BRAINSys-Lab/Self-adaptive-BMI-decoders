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

# First time setup

To download this repository, you can click the green button in the top right and download the entire repo as a zip file. Do not right click to save to file, as that saves the HTML wrapper on the website. Alternatively, you can clone the repository using git. Here's a link to a tutorial teaching you how to set up git yourself: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git. You can also use the code provided below:

```bash
git clone https://github.com/CityU-BRAINSys-Lab/Self-adaptive-BMI-decoders.git
cd Self-adaptive-BMI-decoders
```

This code requires Python 3.11. Install from https://www.python.org/downloads/. Make sure to add Python to your PATH so it can be called by command prompt via the 'py' command. Disabling the path length character limit is not necessary.

To install the libraries this repository needs:

```bash
pip install -r requirements.txt
```

# How to use

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
```

There are several entry points to the program depending on what you want to do:

* `Closed_loop/Closed_loop_models.py` is used when you want to change the models in closed-loop experiments
* `Closed_loop/OPS_closed_loop_training_phase1.py` is used when you want to pre-train the models (the first training stage)
* `Closed_loop/OPS_closed_loop_training_phase2.py` is used when you want to pre-train the models (the second training stage)
* `Open_loop/Open_loop_inference.py` can be used for the inference stage in the open-loop experiment
* `Open_loop/UCSF_open_loop.py` can be used for pre-training in the open-loop experiment
* `Open_loop/Open_loop_models.py` is used when you want to change the models in open-loop experiments
* `utils_files/RL_closed_loop_utils.py` contains several functions used in experiments
* `utils_files/closed_loop_simulator_Shah.py` contains the closed-loop experiments setup functions (adapted from [![Paper](https://img.shields.io/badge/Paper-IOPscience-blue)](https://iopscience.iop.org/article/10.1088/1741-2552/ad1787/meta))  


Inputs for open-loop decoding:  

* The datasets can be downloaded from: https://zenodo.org/records/3854034#.ZCK4eOxBz0o.
* The data pre-processing methods are used in the same way as Neurobench (primate reaching task): https://neurobench.ai/.

Outputs for open-loop decoding:  

* The original label in the dataset is continuous values (velocities). However, we convert it to a discrete value -- discrete classes (see the descriptions in the paper). The functions related to the conversion are listed in the file: `utils_files/RL_closed_loop_utils.py`.

Inputs for closed-loop decoding: 

* The inputs in the closed-loop experiments are the neural signal generated from the Online Prosthesis Simulator (OPS): https://journals.physiology.org/doi/full/10.1152/jn.00503.2010.
* The OPS function and related environment setup are shown in `utils_files/closed_loop_simulator_Shah.py`.

Outputs for closed-loop decoding:

* We can only get discrete values -- class numbers from models (classification tasks), and we need to convert the discrete value to a continuous velocity to update the cursor position of the closed-loop environment. The conversion functions are listed in the `utils_files/RL_closed_loop_utils.py`.


## How to Run

1. Open-loop Decoding  
To train or evaluate the DSNN decoder on standard pre-recorded neural datasets:
```bash
cd Open_loop
python UCSF_open_loop.py  # Training
python Open_loop_inference.py # Inference
```

2. Closed-loop Decoding  
To simulate the self-adaptive continuous learning process (using Banditron/AGREL updates) for closed-loop interactions:
```bash
cd Closed_loop
python OPS_closed_loop_electrode_shift.py  # For electrode shift experiment
python OPS_closed_loop_firing_rate_drift.py  # For firing rate drift experiment
python OPS_closed_loop_loss_neuron.py  # For loss of neurons experiment
```

3. Change device  
The default device is "cuda". We are using the NVIDIA 4090Ti GPU to run it. However, if you want to change a device, feel free to replace "DEVICE" in the code with "cpu".


# Citation
If you find this work, code, or decoders useful for your research, please consider citing our paper:
```text
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
```






