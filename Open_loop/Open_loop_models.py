import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np

import snntorch as snn
from snntorch import surrogate
from utils_files.RL_closed_loop_utils import *

class SNNModelStreamingClassification(nn.Module):
    def __init__(self, input_dim, layer1=65, layer2=40, output_dim=4*2,
                 batch_size=512, sampling_rate=0.01, drop_rate=0.0,
                 beta=0.5, mem_thresh=0.5, spike_grad=surrogate.atan(alpha=2), vel_scale=3.77):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate
        self.vel_scale = vel_scale

        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.bin_window_size = self.batch_size

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)

        self.beta = beta
        self.mem_thresh = mem_thresh
        self.spike_grad = spike_grad
        self.lif1 = snn.Leaky(beta=0.5, spike_grad=self.spike_grad, threshold=0.6, learn_beta=True,
                              learn_threshold=True, init_hidden=True)
        self.lif2 = snn.Leaky(beta=0.6, spike_grad=self.spike_grad, threshold=0.4, learn_beta=True,
                              learn_threshold=True, init_hidden=True)
        self.lif3 = snn.Leaky(beta=0.5, spike_grad=self.spike_grad, threshold=0.6, learn_beta=True,
                              learn_threshold=True, init_hidden=True)

        self.register_buffer("data_buffer", torch.zeros((1, input_dim)).type(torch.float32), persistent=False)
        self.register_buffer("label_buffer", torch.zeros(1, output_dim).type(torch.float32), persistent=False)
        self.input_count = 0
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def reset_mem(self):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()

    def single_forward(self, x):

        cur1 = self.dropout(self.fc1(x))
        spk1 = self.lif1(cur1)

        cur2 = self.dropout(self.fc2(spk1))
        spk2 = self.lif2(cur2)

        cur3 = self.fc3(spk2)
        spk3 = self.lif3(cur3)
        
        pred = torch.cat((self.lif3.mem[:, :int(self.output_dim/2)].clone().t(), self.lif3.mem[:, int(self.output_dim/2):].clone().t()), dim=1)
        return pred

    def forward(self, x):
        predictions = []
        seq_length = x.shape[0]
        for seq in range(seq_length):
            self.input_count += 1
            current_seq = x[seq, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq.unsqueeze(0)), dim=0)
            
            self.data_buffer = self.data_buffer[1:, :]

            spikes = self.data_buffer.clone()

            pred = self.single_forward(spikes)

            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0).squeeze(dim=1)
        return predictions
    
    
class SNNModelStreamingContinuous(nn.Module):
    def __init__(self, input_dim=46, layer1=65, layer2=40, output_dim=4*2,
                 batch_size=512, sampling_rate=0.01, drop_rate=0.0,
                 beta=0.5, mem_thresh=0.5, spike_grad=surrogate.atan(alpha=2), 
                 vel_scale=3.77, error=0, sparse_rate=0, gamma_banditron=0.0001, gamma_agrel=0.001): 
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate
        self.vel_scale = vel_scale

        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.bin_window_size = self.batch_size

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)

        self.beta = beta
        self.mem_thresh = mem_thresh
        self.spike_grad = spike_grad
        self.lif1 = snn.Leaky(beta=0.5, spike_grad=self.spike_grad, threshold=0.6, learn_beta=True,
                              learn_threshold=True, init_hidden=True)
        self.lif2 = snn.Leaky(beta=0.6, spike_grad=self.spike_grad, threshold=0.4, learn_beta=True,
                              learn_threshold=True, init_hidden=True)
        self.lif3 = snn.Leaky(beta=0.5, spike_grad=self.spike_grad, threshold=0.6, learn_beta=True,
                              learn_threshold=True, init_hidden=True)

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)
        self.register_buffer("label_buffer", torch.zeros(1, output_dim).type(torch.float32), persistent=False)
        self.input_count = 0
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.error = error
        self.sparse_rate = sparse_rate
        self.gamma_banditron = gamma_banditron
        self.gamma_agrel = gamma_agrel
        self.update_active = False
        self.update_count = 0
        self.lr_default = 1e-3
        self.lr = self.lr_default
        self.average_reward = []
        self.reset_lr_time_thre = 3 # AGEREL, HRL 2.0, banditron 2.5

        self.criterion = torch.nn.MSELoss()

    def reset_mem(self):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()
        self.update_count = 0
    
    def reset_lr(self):
        self.lr = self.lr_default

    def single_forward(self, x, label_):

        s0 = 1-torch.count_nonzero(x)/torch.numel(x)
        cur1 = self.dropout(self.fc1(x))
        spk1 = self.lif1(cur1)
        s1 = 1-torch.count_nonzero(spk1)/torch.numel(spk1)

        cur2 = self.dropout(self.fc2(spk1))
        spk2 = self.lif2(cur2)
        s2 = 1-torch.count_nonzero(spk2)/torch.numel(spk2)

        cur3 = self.fc3(spk2)
        spk3 = self.lif3(cur3)
        
        pred = torch.cat((self.lif3.mem[:, :int(self.output_dim/2)].clone().t(), self.lif3.mem[:, int(self.output_dim/2):].clone().t()), dim=1)
        
        if self.update_active:
            self.update_count += 1
            pred = self.update_layer_AGREL(input_=x, fc2input_=spk1, fc3input_=spk2, pred_=pred, label_=label_)
            # pred = self.update_layer_banditron(spk2, pred, label_)
                
        return pred, s0, s1, s2

    def forward(self, x, label_):
        predictions = []
        seq_length = x.shape[0]
        sparsity_0 = 0
        sparsity_1 = 0
        sparsity_2 = 0
        for seq in range(seq_length):
            self.input_count += 1
            current_seq = x[seq, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq.unsqueeze(0)), dim=0)

            self.data_buffer = self.data_buffer[1:, :]

            spikes = self.data_buffer.clone()

            pred, s0,s1, s2 = self.single_forward(spikes, label_[seq, :].unsqueeze(0))
            sparsity_0 += s0
            sparsity_1 += s1
            sparsity_2 += s2

            predictions.append(pred)
        predictions = torch.stack(predictions, dim=0)
        return predictions

    
    def update_layer_banditron(self, fc3input_, pred_, label_):
        ################## Banditron #######################
        
        y_hat = torch.argmax(pred_, dim=0)
        
        out_label = torch.zeros_like(pred_)
        for i, item in enumerate(y_hat):
            out_label[item, i] = 1
            
        p = ((1-self.gamma_banditron)*out_label + self.gamma_banditron/(self.output_dim/2)).numpy()
        
        y_tilde = torch.zeros_like(y_hat)
        y_tilde_y = torch.zeros_like(y_hat)
        y_tilde_prob = pred_
        for i in range(label_.shape[1]):
            y_tilde[i] = np.random.choice(range(int(self.output_dim/2)), p=p[:, i])
            y_tilde_y[i] = 1 if y_tilde[i] == label_[:,i] else 0
            y_tilde_prob[y_tilde[i], i] = 1

        sparsify = np.random.choice([True,False],p=[self.sparse_rate,1-self.sparse_rate])
        
        if not sparsify:
            dU = torch.zeros_like(self.fc3.weight)
            for out_order in range(label_.shape[1]):
                for r in range(int(self.output_dim//2)):
        
                    y_tilde_r = 1 if y_tilde[out_order] == r else 0
                    y_hat_r = 1 if y_hat[out_order] == r else 0
                    factor = ((y_tilde_y[out_order]*y_tilde_r)/p[r, out_order] - y_hat_r)

                    if out_order > 0:
                        r_ = r+int(self.output_dim//2)
                    else:
                        r_ = r
                    dU[r_,:] = fc3input_[0, :] * factor
                        
            self.fc3.weight.data = self.fc3.weight.data + dU*self.lr
            
        return y_tilde_prob

    def update_layer_AGREL(self, input_, fc2input_, fc3input_, pred_, label_):
        ################ AGREL #######################
        pred_flat = torch.flatten(pred_.permute(1, 0)).unsqueeze(1)
        y_hat = torch.argmax(pred_, dim=0)
        y_hat[1] = y_hat[1]+int(self.output_dim//2)
        
        
        out_label = torch.zeros((pred_flat.shape[0], 2), dtype=torch.float32, device=input_.device)
        for i, item in enumerate(y_hat):
            out_label[item, i] = 1
        explore = np.random.uniform() < self.gamma_agrel 
        
        # Evaluative framework (refer to the paper to understand the mathematics)
        y_tilde = torch.zeros_like(y_hat, device=input_.device)
        
        if explore:
            y_tilde_prob = pred_
            select_dim = np.random.randint(low=0,high=2)
            y_tilde[select_dim] = np.random.randint(low=0,high=int(self.output_dim//2))  #between 0-8
            y_tilde_prob[y_tilde[select_dim], select_dim] = 1
                
        else:
            y_tilde = y_hat
            y_tilde[1] = y_tilde[1]-int(self.output_dim//2)
            y_tilde_prob = pred_


        sparsify = np.random.choice([True,False],p=[self.sparse_rate,1-self.sparse_rate])
        r = torch.zeros_like(y_hat, device=input_.device)
        delta = torch.zeros_like(y_hat, dtype=torch.float32, device=input_.device)
        if not sparsify:
            for out_order in range(label_.shape[1]):
                if y_tilde[out_order] == label_[:, out_order]:
                    r[out_order] = 1
                    self.average_reward.append(r)
                    if out_order == 1:
                        delta[out_order] = r[out_order] - out_label[y_tilde[out_order]+int(self.output_dim//2), out_order]
                    else:
                        delta[out_order] = r[out_order] - out_label[y_tilde[out_order], out_order]
                    
                else:
                    r[out_order] = 0
                    self.average_reward.append(r)
                    delta[out_order] = np.random.choice([-1 ,1 -  out_label[y_tilde[out_order], out_order].item()],p=[1-self.error,self.error])
            
            
            ### update 3rd layer
            fb3 = (out_label @ delta.unsqueeze(1)).t()
            dW = self.lr*fc3input_.t() @ fb3

            ### update 2nd layer
            g2k = fc3input_
            fb2 = (((out_label) @ delta.unsqueeze(1)).t() @ self.fc3.weight.data)
            dV = self.lr*(fc2input_.t()) @ (g2k*fb2)
            

            ### update 1st layer
            g1j = fc2input_
            fb1 = (g2k * fb2) @ self.fc2.weight.data
            dU = self.lr*input_.t() @ (g1j*fb1) 
            
            self.fc3.weight.data = self.fc3.weight.data + dW.t()
            self.fc2.weight.data = self.fc2.weight.data + dV.t()
            self.fc1.weight.data = self.fc1.weight.data + dU.t()
            

        return y_tilde_prob
    
    
class Banditron(nn.Module):
    def __init__(self, input_dim=46, output_dim=8, error=0, 
                 sparse_rate=0, gamma_banditron=0.0001): 
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.fc1.weight.data.zero_()

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)
        self.input_count = 0

        self.error = error
        self.sparse_rate = sparse_rate
        self.gamma_banditron = gamma_banditron
        self.lr = 5e-3
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.reset_lr_time_thre = 3.0
        self.update_count = 0
        self.update_active = False
        self.bin_window_size = 36 # UCSF: 36; OPS: 1

        
    def reset_lr(self):
        self.lr = 5e-3

    def forward(self, x):
        
        predictions = []
        seq_length = x.shape[0]
        for seq in range(seq_length):
            self.input_count += 1
            current_seq = x[seq, :].unsqueeze(0)
            
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)
            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]

            # Accumulate
            spikes = self.data_buffer.clone()

            acc_spikes = torch.zeros((1, self.input_dim))
            temp = torch.sum(spikes[:, :], dim=0)
            acc_spikes[0, :] = temp
            
            pred_ = self.fc1(acc_spikes.to(x.device))
            pred = torch.cat((pred_[:, :int(self.output_dim/2)].clone(), pred_[:, int(self.output_dim/2):].clone()), dim=0).t()
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=0)
        return predictions
    
    def update_forward(self, x, label_):
        
        predictions = []
        seq_length = x.shape[0]
        for seq in range(seq_length):
            self.input_count += 1
            current_seq = x[seq, :].unsqueeze(0)
            
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)
            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]

            # Accumulate
            spikes = self.data_buffer.clone()

            acc_spikes = torch.zeros((1, self.input_dim))
            temp = torch.sum(spikes[:, :], dim=0)
            acc_spikes[0, :] = temp
            
            pred_ = self.fc1(acc_spikes)
            pred = torch.cat((pred_[:, :int(self.output_dim/2)].clone(), pred_[:, int(self.output_dim/2):].clone()), dim=0).t()
            if self.update_active:
                self.update_count += 1
                pred = self.update_layer_banditron(current_seq, pred, label_[seq, :].unsqueeze(0))
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=0)

        return predictions
    
    def update_layer_banditron(self, input_, pred_, label_):
        ################## Banditron #######################
        y_hat = torch.argmax(pred_, dim=0)

        out_label = torch.zeros_like(pred_)
        for i, item in enumerate(y_hat):
            out_label[item, i] = 1
            
        p = ((1-self.gamma_banditron)*out_label + self.gamma_banditron/(self.output_dim/2)).numpy()
        
        y_tilde = torch.zeros_like(y_hat)
        y_tilde_y = torch.zeros_like(y_hat)
        y_tilde_prob = pred_
        for i in range(label_.shape[1]):
            y_tilde[i] = np.random.choice(range(int(self.output_dim/2)), p=p[:, i])
            y_tilde_y[i] = 1 if y_tilde[i] == label_[:,i] else 0
            y_tilde_prob[y_tilde[i], i] = 1
            

        sparsify = np.random.choice([True,False],p=[self.sparse_rate,1-self.sparse_rate])
        
        if not sparsify:
            dU = torch.zeros_like(self.fc1.weight.data)
            for out_order in range(label_.shape[1]):
                for r in range(int(self.output_dim//2)):
        
                    y_tilde_r = 1 if y_tilde[out_order] == r else 0
                    y_hat_r = 1 if y_hat[out_order] == r else 0
                    factor = ((y_tilde_y[out_order]*y_tilde_r)/p[r, out_order] - y_hat_r)

                    if out_order > 0:
                        r_ = r+int(self.output_dim//2)
                    else:
                        r_ = r
                    dU[r_,:] = input_[0, :] * factor
            
            self.fc1.weight.data = self.fc1.weight.data + dU*self.lr
            
        return y_tilde_prob