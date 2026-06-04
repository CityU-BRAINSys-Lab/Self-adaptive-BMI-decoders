import numpy as np
import csv, copy

from utils_closed_loop.RL_closed_loop_utils import *
from models import *
from utils_closed_loop.closed_loop_simulator_Shah import *
from matplotlib import pyplot as plt

import time
import os
from tqdm import tqdm
import matplotlib.font_manager as font_manager

def get_moving_avg(target_acq_time,window_size):
    acq_time_avg = np.zeros((len(target_acq_time)-window_size))
    for t in range(len(successful_trials)-window_size):
        acq_time_avg[t] = np.mean(target_acq_time[t:t+window_size])
    return acq_time_avg

def match_case(case):
    cases = {
        'SNN': SNNModelStreamingContinuous, 
        # 'ANN': ANNModel3D,
        # 'LSTM': LSTMModel,
        # 'Banditron': Banditron
    }
    # Get the function associated with the case and call it
    if case in cases:
        print(case)
        return cases[case]
    else:
        return "Case not found"
    

def run_trials_training_phase2(model,num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, Lr=0.00001, model_weight_save=True):
    step = 0
    model.train()
    model.update_active=False
    criterion = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.AdamW(model.parameters(), lr=Lr,
                                  betas=(0.9, 0.999), weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=64)
    print("Training phase2")
    for trial in tqdm(range(num_trials)):
        cls.start_trial()
        t = 0
        t_in_range = 0
        
        pos_trial = []
        output_trial = []
        intend_vels = []
        target_lst = []
        start_point_lst = []
        loss_rec = []
        if choice == 'SNN':
            model.reset_mem()
        
            
        while t < max_duration and t_in_range < time_to_target:
            step += 1
            vels,accels = cls.get_velocity()
            pos_trial.append(torch.tensor(copy.deepcopy(cls.position)).unsqueeze(0))
            intend_vels.append(copy.deepcopy(vels))
            target_lst.append(torch.tensor(cls.target).unsqueeze(0))
            start_point_lst.append(torch.tensor([[0,0]]))


            spikes = ops.get_spikes(accels)
            vels = torch.tensor(vels)/time_step
    
            if choice == 'SNN':
                pred_vels = model.forward(torch.tensor(spikes, dtype=torch.float32).unsqueeze(0), magnitude_to_label_class4(vels.unsqueeze(0)/vel_scale, max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4))
                loss_val = criterion(pred_vels, magnitude_to_label_class4((vels.unsqueeze(0)/vel_scale), max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4).long())
                pred_vels = label_to_magnitude_class4(torch.argmax(pred_vels, dim=1), max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4)
                
                loss_rec.append(loss_val.clone().detach())
                
                optimiser.zero_grad()
                loss_val.backward()
                optimiser.step()

                if choice == 'SNN':
                    model.lif1.mem = model.lif1.mem.detach()
                    model.lif2.mem = model.lif2.mem.detach()
                    model.lif3.mem = model.lif3.mem.detach()
                


            output_trial.append(pred_vels.squeeze())
            cls.update_pos(pred_vels.squeeze().detach().numpy()*vel_scale*time_step)
            t,t_in_range = cls.get_times()

            if model_weight_save:
                torch.save(model.state_dict(), model_weight_name_update)
                

        lr_scheduler.step()

        if (t_in_range >= time_to_target):
            successful_trials.append(1.0)
        else:
            successful_trials.append(0.0)

        target_acq_time.append(t)
        print("Time to Target", t*time_step)


        output_trial = np.array(output_trial)
        intend_vels = np.array(intend_vels)
        intend_vels = intend_vels / time_step
        output_trial = output_trial*vel_scale
        
        

        pos_total.append(pos_trial)
        output_total.append(output_trial)
        intend_total.append(intend_vels)

    return pos_total,output_total,intend_total,successful_trials,target_acq_time


if __name__ == "__main__":

    model_name = 'SNN_streaming' 
    DEVICE = 'cuda'

    if 'SNN' in model_name:
        choice = 'SNN'
        learning_rate = 5e-8


    model_weight_name = "./closed_loop_weight/" + "OPS_" + model_name + "_classification" + "_model_state_dict.pth" 
    neurons_save_path = "./closed_loop_weight/" + "OPS_" + model_name + "_classification" + "_neurons.csv" 
    model_weight_name_update = "./closed_loop_weight/" + "Update_OPS_" + model_name + "_classification" + "_model_state_dict.pth" 

    random_seed = 1234
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_seed)
    
    simulation_result_time_to_target = []
    simulation_result_award = []

    time_step = 0.01
    max_duration = int(3/time_step)
    side_radius = 10
    max_vel = 20*time_step #cm/s
    accel_const = 0.3
    min_distance = 8
    target_size = 2.5
    time_to_target = 0.5/time_step


    vel_scale = 3.77
    vel_label_max = max_vel/time_step/vel_scale ## scale velocity according to "A spiking neural network with continuous local learning for robust online brain machine interfaces" paper

    # # ####################################################################################################################################
    ## Training phase2

    cls = CLS(side_radius=side_radius,min_distance=min_distance,max_velocity=max_vel,
            accel_const=accel_const,target_size=target_size)

    neurons = 46
    ops = OPS(neurons,time_step,upper_lmin=5,lower_lmax=40,upper_lmax=100,
            max_accel=accel_const*max_vel,zero_prob=0.5)

    
    if choice == 'SNN':
        model = match_case(choice)(input_dim=neurons)
        model.load_state_dict(torch.load(model_weight_name, map_location=torch.device(DEVICE)))
        model.eval()
        model.reset_mem()
        ops.assign_neurons(neurons_save_path)
    

    num_trials = 30

    successful_trials = []
    target_acq_time = []
    pos_total = []
    output_total = []
    intend_total = []
    error_rate = []

    pos_total,output_total,intend_total,successful_trials,target_acq_time = run_trials_training_phase2(model,num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, Lr=learning_rate, model_weight_save=True)





