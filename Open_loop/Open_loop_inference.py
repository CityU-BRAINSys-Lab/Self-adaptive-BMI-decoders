import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

from tqdm import tqdm
import numpy as np

import snntorch as snn
from snntorch import surrogate

import sys 
sys.path.append("/home/byzhou/desktop/python_code/Self_adaptive decoder/") 

from torch.profiler import profile, record_function, ProfilerActivity
from utils_files.closed_loop_simulator_Shah import *
from Open_loop_models import *
from utils_files.RL_closed_loop_utils import *
from primate_reaching import PrimateReaching

import time
import random
from scipy import signal 
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

import math

DEVICE = "cuda"
BATCH_SIZE = 512

def accuracy_fn(y_true, y_pred):
    y_pred = y_pred.to('cpu')
    y_true = y_true.to('cpu')
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / y_pred.shape[0]) 
    return acc

class MyDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, idx):
        sample = self.samples[idx, :]
        label = self.labels[idx, :]
        return sample, label
    
    def __len__(self):
        return self.samples.shape[0]
        
    
def normalize(vector):
    mag = np.linalg.norm(vector)
    if (mag == 0):
        return vector
    else:
        return vector / mag


def pred_to_vector(pred_label, label):
    # vel =  label_to_magnitude_class6(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # vel =  label_to_magnitude_class8(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    vel =  label_to_magnitude_class4(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # vel =  label_to_magnitude_class12(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # vel =  label_to_magnitude_class16(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # draw_segments(vel, label)
    return vel
    
def convert_vel_to_label(vel):
    # converted_label = magnitude_to_label_class6(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class8(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    converted_label = magnitude_to_label_class4(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class12(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class16(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    return converted_label


def test(test_sample, test_label, test_label_vel, net):
    
    net.eval()
    samples = test_sample
    labels = test_label.long()
    labels_vel = test_label_vel

    test_acc_final = 0
    
    with torch.no_grad():  
        if MODEL_TYPE == "SNN":  
            net.reset_mem()  
        test_r2 = r2()
   
        label = labels.squeeze()
        sample = samples
        label_vel = labels_vel
        
        if model_name == "Banditron":
            pred = net.update_forward(sample, label)
        else:
            pred = net(sample, label)
        pred = torch.argmax(pred, dim=1)
        pred = pred_to_vector(pred, label_vel)
        
        acc = test_r2(net, pred.cpu(), (sample.cpu(), label_vel.cpu()))
        test_acc_final += acc


    return test_acc_final


def data_processing(dataset, samples, labels):
    
    
    samples = samples.t()
    labels = labels.t()
    # converted_label = magnitude_to_label_class6(labels, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class8(labels, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    converted_label = magnitude_to_label_class4(labels, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class12(labels, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class16(labels, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    
    return labels, samples, converted_label
    

if __name__ == "__main__":

    SEED_VALUE=1337
    torch.manual_seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)
    if torch.cuda.is_available():
        print("using cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(SEED_VALUE)

    output_neurons = 4
    model_name = "SNN_streaming" #ANN, SNN_streaming, LSTM, Banditron
    if 'SNN' in model_name:
        MODEL_TYPE = 'SNN'
    else:
        MODEL_TYPE = "ANN"
    model_weight_name = "./model_weights/" + "UCSF_" + model_name + "_classification" + "_model_state_dict_" + str(output_neurons) + ".pth" 
    neurons_save_path = "./model_weights/" + "UCSF_" + model_name + "_classification" + "_neurons.csv" 
    
    
    filenames = ["indy_20161005_06", "indy_20161006_02", "indy_20161007_02", "indy_20161011_03", "indy_20161013_03", "indy_20161014_04"] # "indy_20161005_06", 
    
    for filename in filenames:
    # filename = filenames[2]
        dataset = PrimateReaching(file_path="./datasets/UCSF_dataset/", filename=filename,
                                num_steps=1, train_ratio=0.8, bin_width=0.004,
                                biological_delay=0, max_segment_length=2000, remove_segments_inactive=False) 
        
        if filename == "indy_20161005_06":
            len_samples = dataset.samples.shape[1]
            samples = dataset.samples[:, int(len_samples - len_samples*0.2):]
            labels = dataset.labels[:, int(len_samples - len_samples*0.2):]
            labels, samples, converted_label = data_processing(dataset, samples, labels)
        else:
            labels, samples, converted_label = data_processing(dataset, dataset.samples, dataset.labels)
            len_samples = dataset.samples.shape[1]

        ##### Models
        # net = SNNModelStreamingClassification(input_dim=96)
        net = SNNModelStreamingContinuous(input_dim=96, layer1=30, layer2=30, output_dim=output_neurons*2)
        # net = Banditron(input_dim=96, output_dim=output_neurons*2)
        # net.to(DEVICE)
        
        # if MODEL_TYPE == "SNN":
        net.load_state_dict(torch.load(model_weight_name, weights_only=True, map_location=torch.device(DEVICE)))
        net.update_active = True
        net.lr = 1 # AGREL 0.1/5e-5 Banditron 1/5e-6
        test_acc = test(test_sample=samples, test_label=converted_label, test_label_vel=labels, net=net)
        print("Testing Acc: ", test_acc)