import numpy as np
import time, csv
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

def normalize(vector):
    mag = np.linalg.norm(vector)
    if (mag == 0):
        return vector
    else:
        return vector / mag

class CLS():

    def __init__(self,side_radius=10,min_distance=0,max_velocity=10,accel_const=0.2,target_size=1):
        self.side_radius = side_radius
        self.max_vel = max_velocity
        self.accel_const = accel_const
        self.target = [0.0,0.0]
        self.target_size = target_size
        self.min_distance = min_distance
        self.distance_const = self.max_vel / np.sqrt(2*self.target_size) #Velocity peaks when you're half the side length away from an object

    def start_trial(self):
        target_mag = self.min_distance
        target_angle = np.random.uniform(-np.pi,np.pi)
        # target_angle = np.array([2.1]).item() #-0.4, -1.6, -3, 2.1, 0.75

        self.target = target_mag*np.asarray([np.cos(target_angle),np.sin(target_angle)])
        self.position = np.array([0.0,0.0])
        self.velocity = np.array([0.0,0.0])
        self.t = 0
        self.time_in_range = 0

    def get_velocity(self):
        vector = self.target - self.position
        # print(vector)
        angle = normalize(vector)
        vel_mag = min(self.distance_const*np.sqrt(np.linalg.norm(vector)),self.max_vel)
        new_velocity = vel_mag*angle
        
        delta_velocity = (new_velocity - self.velocity)*self.accel_const
        #The acceleration constant prevents instantaneous jumps in velocity
        # print('delta_velocity', delta_velocity, 'new_velocity', new_velocity, 'self.velocity', self.velocity)

        new_velocity =  self.velocity + delta_velocity
        # print( 'new_velocity', new_velocity)

        return new_velocity,delta_velocity
    

    def update_pos(self,new_vel):
        self.position += new_vel
        self.velocity = new_vel
        self.t += 1
        target_dist = np.linalg.norm(self.position - self.target)
        if (target_dist < self.target_size):
            self.time_in_range += 1
        else:
            self.time_in_range = 0

    def get_times(self):
        return self.t, self.time_in_range

    
class OPS():

    def __init__(self,num_neurons,time_step,upper_lmax,lower_lmax,upper_lmin,max_accel,zero_prob = 0):
        #upper_lmin is the upper bound for the minimum firing rate of a neuron
        #lower_lmax is the lower bound for the maximum firing rate
        #upper_lmax is the upper bound for the maximum firing rate
        
        self.num_neurons = num_neurons
        self.time_step = time_step
        self.max_accel = max_accel
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Synthetic_Neuron(time_step,upper_lmax,lower_lmax,upper_lmin,max_accel,zero_prob=zero_prob))

    def get_spikes(self,v_t):
        spikes= np.zeros((self.num_neurons,))
        for i in range(self.num_neurons):
            spikes[i] = self.neurons[i].get_spike(v_t)
        return spikes

    def remove_neurons(self,indices):
        for i in indices:
            self.neurons[i].removed = True

    def save_neurons(self,filename):
        with open(filename,'w') as file:
            writer = csv.writer(file)
            for i in range(len(self.neurons)):
                neuron = self.neurons[i]
                writer.writerow([neuron.c[0],neuron.c[1],neuron.lambda_min,neuron.lambda_max])

    def assign_neurons(self,filename):
        self.neurons = []
        with open(filename,'r') as file:
            reader = csv.reader(file)
            for row in reader:
                c = np.array([float(row[0]),float(row[1])])
                lambda_min = float(row[2])
                lambda_max = float(row[3])

                neuron = Synthetic_Neuron(self.time_step,upper_lmin=5,lower_lmax=40,upper_lmax=100,max_accel=self.max_accel)
                neuron.assign(c,lambda_min,lambda_max)
                self.neurons.append(neuron)

        self.num_neurons = len(self.neurons)

        
class OPS_rnn():

    def __init__(self,num_neurons,time_step,upper_lmax,lower_lmax,upper_lmin,max_accel,zero_prob = 0):
        #upper_lmin is the upper bound for the minimum firing rate of a neuron
        #lower_lmax is the lower bound for the maximum firing rate
        #upper_lmax is the upper bound for the maximum firing rate
        
        self.num_neurons = num_neurons
        self.time_step = time_step
        self.max_accel = max_accel
        spike_grad=surrogate.atan(alpha=2)
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Synthetic_Neuron_Rate(time_step,upper_lmax,lower_lmax,upper_lmin,max_accel,zero_prob=zero_prob))
        self.lif1 = snn.Leaky(beta=0.98, spike_grad=spike_grad, threshold=1.0, init_hidden=True)
        self.lif2 = snn.Leaky(beta=0.98, spike_grad=spike_grad, threshold=1.0, init_hidden=True)
        self.reset_mem()
        self.W_s = torch.randn((self.num_neurons,self.num_neurons)) * 0.1
        self.W_r = torch.randn((self.num_neurons,self.num_neurons)) * 0.01
    
    def reset_mem(self):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()

    def get_spikes(self,v_t, s_t_1):
        
        
        rate = np.zeros((self.num_neurons,))
        for i in range(self.num_neurons):
            rate[i] = self.neurons[i].get_rate(v_t)
        
        noise =  torch.normal(mean=0, std=0.3, size=(self.num_neurons,))
        rate_t = torch.from_numpy(rate).float()
        # print("Rate before recurrent connection: ", rate_t)
        # rate = self.W_r @ rate_t + self.W_s @ s_t_1

        rate =  rate_t + self.W_s @ s_t_1 + noise
        rate = torch.where(rate > 0, rate, 0) 
        # print("Rate after recurrent connection: ", rate)
        spk_t = self.lif2(rate)
        # print(spk_t)
        
        return np.array(spk_t)
    
    def remove_neurons(self,indices):
        for i in indices:
            self.neurons[i].removed = True

    def save_neurons(self,filename):
        with open(filename,'w') as file:
            writer = csv.writer(file)
            for i in range(len(self.neurons)):
                neuron = self.neurons[i]
                writer.writerow([neuron.c[0],neuron.c[1],neuron.lambda_min,neuron.lambda_max])

    def assign_neurons(self,filename):
        self.neurons = []
        with open(filename,'r') as file:
            reader = csv.reader(file)
            for row in reader:
                c = np.array([float(row[0]),float(row[1])])
                lambda_min = float(row[2])
                lambda_max = float(row[3])

                neuron = Synthetic_Neuron_Rate(self.time_step,upper_lmin=5,lower_lmax=40,upper_lmax=100,max_accel=self.max_accel)
                neuron.assign(c,lambda_min,lambda_max)
                self.neurons.append(neuron)

        self.num_neurons = len(self.neurons)

                
        

class Synthetic_Neuron():
    
    def __init__(self,time_step,upper_lmax,lower_lmax,upper_lmin,max_accel,zero_prob=0):
        self.max_accel = max_accel
        self.time_step = time_step
        
        zero_choice = np.random.choice([0,1],p=[zero_prob,1-zero_prob])
        self.lambda_min = np.random.uniform(0,upper_lmin)*zero_choice

        self.lambda_max = np.random.uniform(max(self.lambda_min,lower_lmax),upper_lmax)

        self.theta_prefer = np.random.uniform(-np.pi,np.pi)
        self.c = np.asarray([np.cos(self.theta_prefer),np.sin(self.theta_prefer)])


        self.removed = False

    def assign(self,c,lambda_min,lambda_max):
        self.c = c
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def get_spike(self,v_t):
        
        noise =  np.random.normal(loc=0,scale=0.3)
        inner_prod = np.clip((1.5*np.inner(self.c,v_t)) / self.max_accel + noise, 0, 1) 
        # inner_prod = min(1,max(0,1.5*np.inner(self.c,v_t))/self.max_accel)
        
        
        lambda_t = (self.lambda_max-self.lambda_min)*inner_prod + self.lambda_min

        p = lambda_t * self.time_step
        if self.removed:
            p = 0

        return np.random.choice([0,1],p=[1-p,p])
    
class Synthetic_Neuron_Rate():
    
    def __init__(self,time_step,upper_lmax,lower_lmax,upper_lmin,max_accel,zero_prob=0):
        self.max_accel = max_accel
        self.time_step = time_step
        
        zero_choice = np.random.choice([0,1],p=[zero_prob,1-zero_prob])
        self.lambda_min = np.random.uniform(0,upper_lmin)*zero_choice

        self.lambda_max = np.random.uniform(max(self.lambda_min,lower_lmax),upper_lmax)

        self.theta_prefer = np.random.uniform(-np.pi,np.pi)
        self.c = np.asarray([np.cos(self.theta_prefer),np.sin(self.theta_prefer)])


        self.removed = False

    def assign(self,c,lambda_min,lambda_max):
        self.c = c
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def get_rate(self,v_t):
        
        inner_prod = np.clip((1.5*np.inner(self.c,v_t)) / self.max_accel, 0, 1) 
        # inner_prod = min(1,max(0,1.5*np.inner(self.c,v_t))/self.max_accel)
        
        
        lambda_t = (self.lambda_max-self.lambda_min)*inner_prod + self.lambda_min

        return np.array(lambda_t*self.time_step)



        

        
        
