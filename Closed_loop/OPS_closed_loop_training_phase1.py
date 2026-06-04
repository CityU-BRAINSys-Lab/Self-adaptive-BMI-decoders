import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

from tqdm import tqdm
import numpy as np

import snntorch as snn
from snntorch import surrogate

from torch.profiler import profile, record_function, ProfilerActivity
from utils_closed_loop.closed_loop_simulator_Shah import *
from models import *
from utils_closed_loop.RL_closed_loop_utils import *

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
    correct_x = torch.eq(y_true[:, 0], y_pred[:, 0]).sum().item() # torch.eq() calculates where two tensors are equal
    correct_y = torch.eq(y_true[:, 1], y_pred[:, 1]).sum().item()
    acc_x = (correct_x / y_pred.shape[0]) 
    acc_y = (correct_y / y_pred.shape[0]) 
    return (acc_x + acc_y)/2

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


def get_correlation(total_output,label):

    r = np.corrcoef([total_output[:,0],label[:,0]])
    r_x = r[0,1]

    r = np.corrcoef([total_output[:,1],label[:,1]])
    r_y = r[0,1]

    return r_x,r_y
    

def get_trial(cls,max_duration):
    cls.start_trial()
    vels = np.zeros((max_duration,2))
    accels = np.zeros((max_duration,2))
    t = 0
    t_in_range = 0
    while t < max_duration and t_in_range < time_to_target:
        vels[t,:],accels[t,:] = cls.get_velocity()
        mag = np.linalg.norm(vels[t,:])
        cls.update_pos(vels[t,:] + np.random.normal(loc=0,scale=0.1*mag,size=(2,)))
        t, t_in_range = cls.get_times()

    vels = vels[:t,:]
    accels = accels[:t,:]
    
    return vels,accels

def get_spikes(ops,accels):
    spikes = np.zeros((len(accels),ops.num_neurons))
    for t in range(len(accels)):
        spikes[t,:] = ops.get_spikes(accels[t])

    return spikes

    
def training(train_sample, train_label, val_sample, val_label, net, model_weight_name):
    epochs = 10
    
    samples = torch.tensor(train_sample, dtype=torch.float32)
    labels = train_label.long()

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=5e-3, 
                                  betas=(0.9, 0.999), weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=64)
    training_set = MyDataset(samples, labels)

    train_loader = DataLoader(
                            dataset=training_set,
                            batch_size=BATCH_SIZE,
                            drop_last=False,
                            shuffle=False,
                        )
    best_training_acc, best_val_acc = float("-inf"), float("-inf")
    current_acc = 0
    
    for epoch in tqdm(range(epochs)):
        net.train()
        for i, (sample, label) in enumerate(train_loader):
            if i % BATCH_SIZE == 0 and MODEL_TYPE == "SNN":
                net.reset_mem()
            
            label = label.to(DEVICE).squeeze()
            sample = sample.to(DEVICE)

            pred = net.forward(sample)
            loss_val = criterion(pred, label)
            pred_acc, ind_acc = torch.max(pred, dim=1) ## Select the class with the highest probability
            acc = accuracy_fn(label, ind_acc)
            
            current_acc = acc
            if current_acc > best_training_acc:
                best_training_acc = current_acc

            optimiser.zero_grad()
            loss_val.backward()
            optimiser.step()
            if MODEL_TYPE == "SNN":
                net.lif1.mem = net.lif1.mem.detach()
                net.lif2.mem = net.lif2.mem.detach()
                net.lif3.mem = net.lif3.mem.detach()

        print(f"{epoch} training Acc: {current_acc}")
        lr_scheduler.step()

        current_val_acc = validation(val_sample, val_label, net)
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            model_weight_name = model_weight_name
            torch.save(net.state_dict(), model_weight_name)
        print(f"{epoch} validation Acc Score: {current_val_acc}")
            
def validation(val_sample, val_label, net):
    net.eval()
    net.to(DEVICE)
    samples = torch.tensor(val_sample, dtype=torch.float32)
    labels = val_label.long()
    
    val_acc_final = 0
    with torch.no_grad():
        if MODEL_TYPE == "SNN":
            net.reset_mem()
        for i in range(0, samples.shape[0]-1, 1):
            
            label = labels[i:i+1, :].to(DEVICE)
            sample = samples[i:i+1, :].to(DEVICE)

            pred = net(sample)
            pred_acc, ind_acc = torch.max(pred, dim=1)
            acc = accuracy_fn(label, ind_acc)

            val_acc_final += acc
        

    return val_acc_final/(i+1)


def test(test_sample, test_label, net):
    
    net.load_state_dict(torch.load(model_weight_name, weights_only=True, map_location=torch.device(DEVICE)))
    net.eval()
    samples = torch.tensor(test_sample, dtype=torch.float32)
    labels = test_label.long()

    test_acc_final = 0
    with torch.no_grad():   
        if MODEL_TYPE == "SNN": 
            net.reset_mem() 
   
        label = labels.to(DEVICE).squeeze()
        sample = samples.to(DEVICE)

        pred = net(sample)

        pred_acc, ind_acc = torch.max(pred, dim=1)

        acc = accuracy_fn(label, ind_acc)

        test_acc_final += acc

    return test_acc_final

def split_data(train_ratio, vels_train, vels_test, train_sample, test_sample):
    train_len = math.floor(train_ratio * len(vels_train))
    val_len = len(vels_train) - train_len

    train_label = vels_train[:train_len,:]
    val_sample = train_sample[train_len:,:]
    train_sample = train_sample[:train_len,:]
    val_label = vels_train[train_len:,:]
    test_label = vels_test

    return train_sample, train_label, val_sample, val_label, test_sample, test_label

def convert_vel_to_label(vel, max_vel):
    converted_label = magnitude_to_label_class4(vel, max_mag=max_vel, min_mag=-max_vel, label_number=4)
    return converted_label


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


    model_name = "SNN_streaming" 
    print("Model Name: ", model_name)
    if 'SNN' in model_name:
        MODEL_TYPE = "SNN"
    else:
        MODEL_TYPE = "ANN"
    model_weight_name = "./closed_loop_weight/" + "OPS_" + model_name + "_classification" + "_model_state_dict.pth" 
    neurons_save_path = "./closed_loop_weight/" + "OPS_" + model_name + "_classification" + "_neurons.csv" 
    
    time_step = 0.01
    max_duration = int(3/time_step)
    side_radius = 10
    max_vel = 20*time_step #cm/s
    accel_const = 0.3
    min_distance = 8
    target_size = 2.5
    time_to_target = 0.5/time_step

    vel_scale = 3.77

    train_trials = 400
    test_trials = 100

    cls = CLS(side_radius=side_radius,min_distance=min_distance,max_velocity=max_vel,
            accel_const=accel_const,target_size=target_size)

    neurons = 46 
    ops = OPS(neurons,time_step,upper_lmin=5,lower_lmax=40,upper_lmax=100,
            max_accel=accel_const*max_vel,zero_prob=0.5)
    ops.save_neurons(neurons_save_path)
    

    vels_train = np.zeros((1,2))
    spikes_train = np.zeros((1,neurons))
    vels_test = np.zeros((1,2))
    spikes_test = np.zeros((1,neurons))

    for trial in range(train_trials+1):
        ops.assign_neurons(neurons_save_path)
        vels,accels = get_trial(cls,max_duration)
        vels_train = np.concatenate((vels_train,vels),axis=0)
        spikes = get_spikes(ops,accels)
        spikes_train = np.concatenate((spikes_train,spikes),axis=0)
        
    for trial in range(test_trials+1):
        ops.assign_neurons(neurons_save_path)
        vels,accels = get_trial(cls,max_duration)
        vels_test = np.concatenate((vels_test,vels),axis=0)
        spikes = get_spikes(ops,accels)
        spikes_test = np.concatenate((spikes_test,spikes),axis=0)
    
    vels_train = vels_train / time_step / vel_scale
    vels_test = vels_test / time_step / vel_scale
    vel_label_max = max_vel/time_step/vel_scale ## scale velocity according to "A spiking neural network with continuous local learning for robust online brain machine interfaces" paper
    
    converted_train_label = convert_vel_to_label(vels_train, vel_label_max)
    converted_test_label = convert_vel_to_label(vels_test, vel_label_max)

    train_sample, train_label, val_sample, val_label, test_sample, test_label = split_data(0.8, converted_train_label, converted_test_label, spikes_train, spikes_test) ## split training and validation data

    if 'SNN' in model_name:
        net = SNNModelStreamingClassification(input_dim=neurons, drop_rate=0.3)

    net.to(DEVICE)

    print("Start training")
    training(train_sample, train_label, val_sample, val_label, net, model_weight_name)
    test_acc = test(test_sample, test_label, net)
    print("Testing Acc: ", test_acc)



