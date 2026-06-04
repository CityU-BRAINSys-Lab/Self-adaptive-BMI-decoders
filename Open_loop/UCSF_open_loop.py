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
    def __init__(self, samples, labels, labels_vel):
        self.samples = samples
        self.labels = labels
        self.labels_vel = labels_vel

    def __getitem__(self, idx):
        sample = self.samples[idx, :]
        label = self.labels[idx, :]
        label_vel = self.labels_vel[idx, :]
        return sample, label, label_vel
    
    def __len__(self):
        return self.samples.shape[0]
        
    
def normalize(vector):
    mag = np.linalg.norm(vector)
    if (mag == 0):
        return vector
    else:
        return vector / mag


def pred_to_vector(pred_label, label):
    vel =  label_to_magnitude_class4(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # vel =  label_to_magnitude_class6(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # vel =  label_to_magnitude_class8(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # vel =  label_to_magnitude_class12(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # vel =  label_to_magnitude_class16(pred_label, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # draw_segments(vel, label)
    return vel
    
def convert_vel_to_label(vel):
    converted_label = magnitude_to_label_class4(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class6(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class8(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class12(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    # converted_label = magnitude_to_label_class16(vel, max_mag=0.5, min_mag=-0.5, label_number=output_neurons)
    return converted_label
        
    
def training(train_samples, train_label, train_labels_vel, val_samples, val_label, val_labels_vel, net, model_weight_name):
    epochs = 50
    
    samples = train_samples
    labels = train_label.long()
    labels_vel = train_labels_vel
    
    training_set = MyDataset(samples, labels, labels_vel)

    train_loader = DataLoader(
                            dataset=training_set,
                            batch_size=BATCH_SIZE,
                            drop_last=False,
                            shuffle=False,
                        )

    # criterion = torch.nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=0.01, # 0.01 SNN, 5e-5 Bandiron
                                  betas=(0.9, 0.999), weight_decay=0.05) # 0.05 SNN
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=64)
    best_training_acc, best_val_acc = float("-inf"), float("-inf")
    current_acc = 0
    
    for epoch in tqdm(range(epochs)):
        net.train()
        training_r2 = r2()
        for i, (sample, label, label_vel) in enumerate(train_loader):
            if i % BATCH_SIZE == 0 and MODEL_TYPE == "SNN":
                net.reset_mem()
            
            label = label.to(DEVICE).squeeze()
            sample = sample.to(DEVICE)
            label_vel = label_vel.to(DEVICE)
            
            pred = net(sample)
            loss_val = criterion(pred, label)
           
            if current_acc > best_training_acc:
                best_training_acc = current_acc

            optimiser.zero_grad()
            loss_val.backward()
            optimiser.step()

            if MODEL_TYPE == "SNN":
                net.lif1.mem = net.lif1.mem.detach()
                net.lif2.mem = net.lif2.mem.detach()
                net.lif3.mem = net.lif3.mem.detach()

        pred = torch.argmax(pred, dim=1)
        pred = pred_to_vector(pred, label_vel)
        current_acc  = training_r2(net, pred.cpu(), (sample.cpu(), label_vel.cpu()))
        print(f"{epoch} training acc: {current_acc}")
        lr_scheduler.step()

        current_val_acc = validation(val_samples, val_label, val_labels_vel, net)
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(net.state_dict(), model_weight_name)
        print(f"{epoch} validation acc: {current_val_acc}")
    
    print(f"Best Validation R2 is: {best_val_acc}")
            
def validation(val_sample, val_label, val_labels_vel, net):
    net.eval()
    net.to(DEVICE)
    samples = val_sample
    labels = val_label.long()
    labels_vel = val_labels_vel
    
    val_acc_final = 0
    with torch.no_grad():
        if MODEL_TYPE == "SNN":
            net.reset_mem()
        val_r2 = r2()
   
        label = labels.to(DEVICE).squeeze()
        sample = samples.to(DEVICE)
        label_vel = labels_vel.to(DEVICE)

        pred = net(sample)
        
        pred = torch.argmax(pred, dim=1)
        pred = pred_to_vector(pred, label_vel)
        acc = val_r2(net, pred.cpu(), (sample.cpu(), label_vel.cpu()))
        
        val_acc_final += acc
        

    return val_acc_final

def test(test_sample, test_label, test_label_vel, net):
    
    net.load_state_dict(torch.load(model_weight_name, weights_only=True, map_location=torch.device(DEVICE)))
    net.eval()
    samples = test_sample
    labels = test_label.long()
    labels_vel = test_label_vel

    test_acc_final = 0
    with torch.no_grad():    
        if MODEL_TYPE == "SNN":
            net.reset_mem()  
        test_r2 = r2()
   
        label = labels.to(DEVICE).squeeze()
        sample = samples.to(DEVICE)
        label_vel = labels_vel.to(DEVICE)

        pred = net(sample)
        pred = torch.argmax(pred, dim=1)
        pred = pred_to_vector(pred, label_vel)
        
        acc = test_r2(net, pred.cpu(), (sample.cpu(), label_vel.cpu()))
        test_acc_final += acc

    return test_acc_final


def data_processing(dataset):
    train_samples, train_labels = [], []
    for segment in dataset.time_segments:
        if segment[0] in dataset.ind_train and (segment[1] - segment[0]) <= dataset.max_segment_length:
            train_samples.append(dataset.samples[:, segment[0]:segment[1]])
            train_labels.append(dataset.labels[:, segment[0]:segment[1]])
    train_samples = torch.cat(train_samples, dim=1).t()
    train_labels = torch.cat(train_labels, dim=1).t()
            
    val_samples, val_labels = [], []
    for segment in dataset.time_segments:
        if segment[0] in dataset.ind_val:
            val_samples.append(dataset.samples[:, segment[0]:segment[1]])
            val_labels.append(dataset.labels[:, segment[0]:segment[1]])
    val_samples = torch.cat(val_samples, dim=1).t()
    val_labels = torch.cat(val_labels, dim=1).t()
    
    test_samples, test_labels = [], []
    for segment in dataset.time_segments:
        if segment[0] in dataset.ind_test:
            test_samples.append(dataset.samples[:, segment[0]:segment[1]])
            test_labels.append(dataset.labels[:, segment[0]:segment[1]])
    test_samples = torch.cat(test_samples, dim=1).t()
    test_labels = torch.cat(test_labels, dim=1).t()
    
    converted_train_label = convert_vel_to_label(train_labels)
    converted_val_label = convert_vel_to_label(val_labels)
    converted_test_label = convert_vel_to_label(test_labels)
    
    return train_labels, val_labels, test_labels, train_samples, val_samples, test_samples, converted_train_label, converted_val_label, converted_test_label

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

    output_neurons = 16
    model_name = "SNN_streaming" #ANN, SNN_streaming, LSTM, Banditron
    print("Model Name: ", model_name)
    if 'SNN' in model_name:
        MODEL_TYPE = "SNN"
    else:
        MODEL_TYPE = "ANN"

    model_weight_name = "./model_weights/" + "UCSF_" + model_name + "_classification" + "_model_state_dict_" + str(output_neurons) + ".pth" 
    neurons_save_path = "./model_weights/" + "UCSF_" + model_name + "_classification" + "_neurons.csv" 
    

    
    filename = "indy_20161005_06"
    layer1=30
    layer2=30
    

    dataset = PrimateReaching(file_path="./datasets/UCSF_dataset/", filename=filename,
                            num_steps=1, train_ratio=0.8, bin_width=0.004,
                            biological_delay=0, max_segment_length=2000, remove_segments_inactive=False) 
    
    
    train_labels_vel, val_labels_vel, test_labels_vel, train_samples, val_samples, test_samples, converted_train_label, converted_val_label, converted_test_label = data_processing(dataset)

    if 'SNN' in model_name:
        net = SNNModelStreamingClassification(input_dim=96, drop_rate=0.1, layer1=layer1, layer2=layer2, output_dim=output_neurons*2)
    elif 'Banditron' in model_name:
        net = Banditron(input_dim=96, output_dim=output_neurons*2)
    
    net.to(DEVICE)

    print("Start training")
    training(train_samples, converted_train_label, train_labels_vel, val_samples, converted_val_label, val_labels_vel, net, model_weight_name)
    
    test_acc = test(test_samples, converted_test_label, test_labels_vel, net)
    print("Testing Acc: ", test_acc)
    
    
    
    



