import numpy as np
import csv, copy

import sys 
sys.path.append("/home/byzhou/desktop/python_code/Self_adaptive decoder/") 

from utils_files.RL_closed_loop_utils import *
from Closed_loop_models import *
from utils_files.closed_loop_simulator_Shah import *
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

def save_results(model_name, rewards, time_to_target, path):

    with open(os.path.join(path, "trials_results.txt"), "a") as f:
        f.write("model_name" + str(model_name) + "\n")

        f.write("average_time_to_target" + "\n")
        f.write("[")
        f.write(','.join(str(i) for i in time_to_target))
        f.write("]")
        f.write("\n")

        f.write("average_rewards" + "\n")
        f.write("[")
        f.write(','.join(str(i) for i in rewards))
        f.write("]")
        f.write("\n")

        f.close()
        
def firing_rate_drift(neurons_select, target_lambda_max, neurons):
    for i, neuron_id in enumerate(neurons_select):
        target_lambda_max_ = target_lambda_max[i]
        time_constant = np.random.normal(30,1)
        current_lambda_max = neurons[neuron_id].lambda_max

        if current_lambda_max <= target_lambda_max_:
            continue
        else:
            current_lambda_max = math.exp(-(1/time_constant)) * current_lambda_max
            neurons[neuron_id].lambda_max = current_lambda_max
    return neurons


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
    

def run_trials(num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, model_weight_save=False, target_lambda_max=[]):
    step = 0
    lr_reset_count = 0
    for trial in tqdm(range(num_trials)):
        cls.start_trial()
        t = 0
        t_in_range = 0
        lr_reset_count += 1
        
        pos_trial = []
        output_trial = []
        intend_vels = []
        target_lst = []
        start_point_lst = []
        
        if choice == 'SNN':
            model.reset_mem()
            
        if stage == 2:
            ops.neurons = firing_rate_drift(neurons_select, target_lambda_max, ops.neurons)

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
                pred_vels = model.forward(torch.tensor(spikes, dtype=torch.float32).unsqueeze(0), magnitude_to_label_class4(vels.unsqueeze(0)/vel_scale, max_mag=vel_label_max, min_mag=-vel_label_max, label_number=int(output_neurons//2)))
                pred_vels = label_to_magnitude_class4(torch.argmax(pred_vels, dim=1), max_mag=vel_label_max, min_mag=-vel_label_max, label_number=int(output_neurons//2))  ## from label to velocity
            
            output_trial.append(pred_vels.squeeze())
            cls.update_pos(pred_vels.squeeze().detach().numpy()*vel_scale*time_step) # update position in simulator
            t,t_in_range = cls.get_times()


        if (t_in_range >= time_to_target):
            successful_trials.append(1.0)
        else:
            successful_trials.append(0.0)

        target_acq_time.append(t)
        print("Trials Number", trial, "Time to Target", t*time_step)
        
        if model_weight_save:
            torch.save(model.state_dict(), model_weight_name_update)

        pos_trial = np.array(pos_trial)
        output_trial = np.array(output_trial)
        intend_vels = np.array(intend_vels)
        intend_vels = intend_vels / time_step
        output_trial = output_trial*vel_scale
        
        
        pos_total.append(pos_trial)
        output_total.append(output_trial)
        intend_total.append(intend_vels)


        if target_acq_time[-1]*time_step >= model.reset_lr_time_thre:
            model.reset_lr()
            lr_reset_count = 0
        
    return pos_total,output_total,intend_total,successful_trials,target_acq_time


if __name__ == "__main__":

    model_name = 'SNN_streaming' # SNN_streaming, ANN, LSTM, Banditron
    DEVICE = 'cuda'

    if 'ANN' in model_name:
        choice = 'ANN'
    elif 'SNN' in model_name:
        choice = 'SNN'


    model_weight_name = "./model_weights/" + "OPS_" + model_name + "_classification" + "_model_state_dict.pth" 
    neurons_save_path = "./model_weights/" + "OPS_" + model_name + "_classification" + "_neurons.csv" 
    model_weight_name_update = "./model_weights/" + "Update_OPS_" + model_name + "_classification" + "_model_state_dict.pth" 
    
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
    vel_label_max = max_vel/time_step/vel_scale
    perturb_ratio = 0.6
    
    ###################################### normal closed_loop

    for sim_num in range(10):

        cls = CLS(side_radius=side_radius,min_distance=min_distance,max_velocity=max_vel,
            accel_const=accel_const,target_size=target_size)

        neurons = 46
        ops = OPS(neurons,time_step,upper_lmin=5,lower_lmax=40,upper_lmax=100,
                max_accel=accel_const*max_vel,zero_prob=0.5)

        output_neurons = 8

        
        if choice == 'SNN':
            model = match_case(choice)(input_dim=neurons)
            model.load_state_dict(torch.load(model_weight_name_update, map_location=torch.device(DEVICE)), strict=False)
            model.eval()
            model.reset_mem()
            ops.assign_neurons(neurons_save_path)
        
        
        ###################################### normal closed_loop
        num_trials = 50

        successful_trials = []
        target_acq_time = []
        pos_total = []
        output_total = []
        intend_total = []
        error_rate = []
        
        model.update_active=False
        stage = 1
        pos_total,output_total,intend_total,successful_trials,target_acq_time = run_trials(num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, target_lambda_max=[])

        ####################################### Firing rate Drift

        neurons_remaining = list(np.arange(neurons))
        num_select = int(neurons*perturb_ratio) # 30
        neurons_select = np.random.choice(neurons_remaining, num_select, replace=False)

        target_lambda_max = []
        for neuron_id in neurons_select:
            target_lambda_max.append(np.random.uniform(0, 30))

        num_trials = 100

        model.update_active=True
        stage = 2
        pos_total,output_total,intend_total,successful_trials,target_acq_time = run_trials(num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, target_lambda_max=target_lambda_max)


        simulation_result_time_to_target.append([i*time_step for i in target_acq_time])
        simulation_result_award.append(successful_trials)


    ave_rewards = np.mean(np.array(simulation_result_award), axis=0)
    ave_time_to_target = np.mean(np.array(simulation_result_time_to_target), axis=0)
    ave_time_to_target = get_moving_avg(ave_time_to_target,4)
    

    save_results(model_name=match_case(choice), rewards=ave_rewards, time_to_target=ave_time_to_target, path="./Closed_loop/Results/")




