import numpy as np
import csv, copy

from utils_closed_loop.RL_closed_loop_utils import *
from RL_closed_loop_models import *
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

def draw_trajectory_figure(pos_trial, target_lst, start_point_lst):
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(111)
    
    x = [i*0.004 for i in range(len(pos_trial))]
    plt.plot(torch.cat(pos_trial)[:, 0], torch.cat(pos_trial)[:, 1], color="darkgoldenrod", linewidth=5.5, label="Pred Trajectory")
    # rect1 = patches.Rectangle((target_zone[2,0], target_zone[2,1]), 7.5, 7.5, color='b', fill=False)
    # ax1.add_patch(rect1)
    circle = plt.Circle((torch.cat(target_lst)[0, 0], torch.cat(target_lst)[0, 1]), target_size, color='#D4E7E9', alpha=0.7)
    ax1.set_xlim(-side_radius, side_radius)
    ax1.set_ylim(-side_radius, side_radius)
    ax1.add_patch(circle)
    plt.scatter(torch.cat(target_lst)[:, 0], torch.cat(target_lst)[:, 1], color="forestgreen", s=300, label="Target")
    plt.scatter(torch.cat(start_point_lst)[:, 0], torch.cat(start_point_lst)[:, 1], color="#356a9f", s=300, label="Start Point")
    plt.xlabel("x/mm", fontsize=30, fontweight='bold')
    plt.ylabel("y/mm", fontsize=30, fontweight='bold')
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    legfont = font_manager.FontProperties(family= 'Times new roman',  
                   weight='bold',
                   style='normal', size=30)
    plt.legend(loc='best', prop=legfont)
    ax1.set_title('Trajectory', fontsize=40, fontweight='bold')

    # ax2 = fig.add_subplot(212)
    # plt.plot(x, torch.cat(pos_trial)[:, 1], color='r', linewidth=1.5, label="velocity")
    # plt.xlabel("time/s", fontsize=20, fontweight='bold')
    # plt.ylabel("y_velocity (mm/s)", fontsize=20, fontweight='bold')
    # plt.legend(loc="best")
    # plt.xticks(fontsize=16, fontweight='bold')
    # plt.yticks(fontsize=16, fontweight='bold')   
    # ax2.set_title("Velocity", fontsize=20, fontweight='bold')
    # plt.savefig("./Figures/{}.jpg".format(step), dpi=300)
    plt.savefig("./test.jpg", dpi=300)
    plt.close()

def save_results(model_name, rewards, time_to_target, path):

    with open(os.path.join(path, "test_data.txt"), "a") as f:
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



def save_results_trajectory(model_name, pos_trial, target_lst, start_point_lst, path):

    with open(os.path.join(path, "test_data.txt"), "a") as f:
        f.write("model_name" + str(model_name) + "\n")

        f.write("pos_trial" + "\n")
        f.write("[")
        for i in range(pos_trial.shape[0]):
            f.write("[")
            for j in range(pos_trial.shape[1]):
                f.write(str(pos_trial[i,j].item()))
                if j == 0:
                    f.write(',')
            f.write("]")
            f.write(',')
        f.write("]")
        f.write("\n")

        f.write("target_lst" + "\n")
        f.write("[")
        for i in range(target_lst.shape[0]):
            f.write("[")
            for j in range(target_lst.shape[1]):
                f.write(str(target_lst[i,j].item()))
                if j == 0:
                    f.write(',')
            f.write("]")
            f.write(',')
        f.write("]")
        f.write("\n")

        f.write("start_point_lst" + "\n")
        f.write("[")
        for i in range(start_point_lst.shape[0]):
            f.write("[")
            for j in range(start_point_lst.shape[1]):
                f.write(str(start_point_lst[i,j].item()))
                if j == 0:
                    f.write(',')
            f.write("]")
            f.write(',')
        f.write("]")
        f.write("\n")


        f.close()


def match_case(case):
    cases = {
        'SNN': SNNModelStreamingContinuous, # SNNModelStreamingContinuous, SNNModelStreamingClassification
        'ANN': ANNModel3D,
        'LSTM': LSTMModel,
        'Banditron': Banditron
    }
    # Get the function associated with the case and call it
    if case in cases:
        print(case)
        return cases[case]
    else:
        return "Case not found"
    

def run_trials(num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, model_weight_save=False):
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
        vel_mag = []
        if choice == 'SNN':
            model.reset_mem()
        elif choice == 'ANN':
            model.data_buffer = torch.zeros(1, neurons).type(torch.float32)
        elif choice == 'LSTM':
            model.reset_hidden_state_zero()
    
            
        while t < max_duration and t_in_range < time_to_target:
            step += 1
            vels,accels = cls.get_velocity()
            pos_trial.append(torch.tensor(copy.deepcopy(cls.position)).unsqueeze(0))
            intend_vels.append(copy.deepcopy(vels))
            target_lst.append(torch.tensor(cls.target).unsqueeze(0))
            start_point_lst.append(torch.tensor([[0,0]]))


            spikes = ops.get_spikes(accels)
            vels = torch.tensor(vels)/time_step
    
            if choice == 'SNN' or choice =='ANN' or choice =='LSTM' or choice =='Banditron':
                pred_vels = model.forward(torch.tensor(spikes, dtype=torch.float32).unsqueeze(0), magnitude_to_label_class4(vels.unsqueeze(0)/vel_scale, max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4))
                pred_vels = label_to_magnitude_class4(torch.argmax(pred_vels, dim=1), max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4)
            
            output_trial.append(pred_vels.squeeze())
            cls.update_pos(pred_vels.squeeze().detach().numpy()*vel_scale*time_step)
            t,t_in_range = cls.get_times()

            # draw_trajectory_figure(pos_trial, target_lst, start_point_lst)


        if (t_in_range >= time_to_target):
            successful_trials.append(1.0)
        else:
            successful_trials.append(0.0)

        target_acq_time.append(t)
        print("Time to Target", t*time_step)
        
        if model_weight_save:
            torch.save(model.state_dict(), model_weight_name_update)

        # save_results_trajectory(model_name=match_case(choice), pos_trial=torch.cat(pos_trial), target_lst=torch.cat(target_lst), start_point_lst=torch.cat(start_point_lst), path="./Results")

        pos_trial = np.array(pos_trial)
        output_trial = np.array(output_trial)
        intend_vels = np.array(intend_vels)
        intend_vels = intend_vels / time_step
        output_trial = output_trial*vel_scale
        
        
        pos_total.append(pos_trial)
        output_total.append(output_trial)
        intend_total.append(intend_vels)

        # if (lr_reset_count+1) % 30 == 0 and stage == 2:
        #     # model.update_active=False
        #     model.lr = model.lr - model.lr*0.1

        if target_acq_time[-1]*time_step >= model.reset_lr_time_thre:
            model.reset_lr()
            lr_reset_count = 0
        

    return pos_total,output_total,intend_total,successful_trials,target_acq_time


def run_trials_training_phase2(model,num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, Lr=0.00001, model_weight_save=True):
    step = 0
    model.train()
    model.update_active=False
    # criterion = torch.nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.AdamW(model.parameters(), lr=Lr, # SNN 5e-8, Banditron 5e-3
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
        elif choice == 'ANN':
            model.data_buffer = torch.zeros(1, neurons).type(torch.float32)
        elif choice == 'LSTM':
            model.reset_hidden_state_zero()
        
            
        while t < max_duration and t_in_range < time_to_target:
            step += 1
            vels,accels = cls.get_velocity()
            pos_trial.append(torch.tensor(copy.deepcopy(cls.position)).unsqueeze(0))
            intend_vels.append(copy.deepcopy(vels))
            target_lst.append(torch.tensor(cls.target).unsqueeze(0))
            start_point_lst.append(torch.tensor([[0,0]]))


            spikes = ops.get_spikes(accels)
            vels = torch.tensor(vels)/time_step
    
            if choice == 'SNN' or choice =='ANN' or choice =='LSTM' or choice=='Banditron':
                pred_vels = model.forward(torch.tensor(spikes, dtype=torch.float32).unsqueeze(0), magnitude_to_label_class4(vels.unsqueeze(0)/vel_scale, max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4))
                loss_val = criterion(pred_vels, magnitude_to_label_class4((vels.unsqueeze(0)/vel_scale), max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4).long())
                pred_vels = label_to_magnitude_class4(torch.argmax(pred_vels, dim=1), max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4)
                
                # draw_trajectory_figure(pos_trial, target_lst, start_point_lst)
                loss_rec.append(loss_val.clone().detach())
                
                optimiser.zero_grad()
                loss_val.backward()
                optimiser.step()

                if choice == 'SNN':
                    model.lif1.mem = model.lif1.mem.detach()
                    model.lif2.mem = model.lif2.mem.detach()
                    model.lif3.mem = model.lif3.mem.detach()
                    
            # elif choice=='Banditron':
            #     pred_vels = model.forward(torch.tensor(spikes, dtype=torch.float32).unsqueeze(0), magnitude_to_label_class4(vels.unsqueeze(0)/vel_scale, max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4))
            #     pred_vels = label_to_magnitude_class4(torch.argmax(pred_vels, dim=1), max_mag=vel_label_max, min_mag=-vel_label_max, label_number=4)
            #     model.lr_default = 1e-3
            #     model.update_active=True
                


            output_trial.append(pred_vels.squeeze())
            cls.update_pos(pred_vels.squeeze().detach().numpy()*vel_scale*time_step)
            t,t_in_range = cls.get_times()

            if model_weight_save:
                torch.save(model.state_dict(), model_weight_name_update)
                
        if choice == 'SNN' or choice =='ANN' or choice =='LSTM':
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

    model_name = 'Banditron' # SNN_streaming, ANN, LSTM, Banditron
    DEVICE = 'cuda'

    # Lr: SNN 5e-8, Banditron 5e-3
    if 'ANN' in model_name:
        choice = 'ANN'
        learning_rate = 5e-3
    elif 'SNN' in model_name:
        choice = 'SNN'
        learning_rate = 5e-8
    elif 'LSTM' in model_name:
        choice = 'LSTM'
        learning_rate = 5e-3
    elif 'Banditron' in model_name:
        choice = 'Banditron'
        learning_rate = 5e-10

    model_weight_name = "./SNN_closed_loop/open_loop_50/" + "OPS_" + model_name + "_classification" + "_model_state_dict.pth" 
    neurons_save_path = "./SNN_closed_loop/open_loop_50/" + "OPS_" + model_name + "_classification" + "_neurons.csv" 
    model_weight_name_update = "./SNN_closed_loop/open_loop_50/" + "Update_OPS_" + model_name + "_classification" + "_model_state_dict.pth" 
    # model_weight_name = "./SNN_closed_loop/open_loop_50/" + "OPS_" + model_name + "_classification" + "_model_state_dict.pth" 
    # neurons_save_path = "./SNN_closed_loop/open_loop_50/" + "OPS_" + model_name + "_classification" + "_neurons.csv" 
    # model_weight_name_update = "./SNN_closed_loop/open_loop_50/" + "Update_OPS_" + model_name + "_classification" + "_model_state_dict.pth" 

    random_seed = 1234 #5678
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

    # # ####################################################################################################################################
    ## Training phase2

    cls = CLS(side_radius=side_radius,min_distance=min_distance,max_velocity=max_vel,
            accel_const=accel_const,target_size=target_size)

    neurons = 46
    ops = OPS(neurons,time_step,upper_lmin=5,lower_lmax=40,upper_lmax=100,
            max_accel=accel_const*max_vel,zero_prob=0.5)
    
    # ops.save_neurons(neurons_save_path)
    
    if choice == 'SNN':
        model = match_case(choice)(input_dim=neurons)
        model.load_state_dict(torch.load(model_weight_name, map_location=torch.device(DEVICE)))
        model.eval()
        model.reset_mem()
        ops.assign_neurons(neurons_save_path)
    elif choice == 'ANN':
        model = match_case(choice)(input_dim=neurons)
        model.load_state_dict(torch.load(model_weight_name, map_location=torch.device(DEVICE)))
        model.eval()
        ops.assign_neurons(neurons_save_path)
    elif choice == 'LSTM':
        model = match_case(choice)(input_dim=neurons)
        model.load_state_dict(torch.load(model_weight_name, map_location=torch.device(DEVICE)))
        model.eval()
        model.reset_hidden_state_zero()
        ops.assign_neurons(neurons_save_path)
    else:
        model = match_case(choice)(input_dim=neurons)
        model.load_state_dict(torch.load(model_weight_name, map_location=torch.device(DEVICE)))
        model.eval()
        ops.assign_neurons(neurons_save_path)

    num_trials = 30

    successful_trials = []
    target_acq_time = []
    pos_total = []
    output_total = []
    intend_total = []
    error_rate = []
    
    # model.update_active=True
    # model.update_active=False
    # stage = 0

    pos_total,output_total,intend_total,successful_trials,target_acq_time = run_trials_training_phase2(model,num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, Lr=learning_rate, model_weight_save=True)
    # pos_total,output_total,intend_total,successful_trials,target_acq_time = run_trials(num_trials,pos_total,output_total,intend_total,successful_trials,target_acq_time, model_weight_save=True)




