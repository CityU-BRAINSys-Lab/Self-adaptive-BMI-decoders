import scipy.special as sp
import numpy as np
import torch
import math
import os

def sigmoid(x):
  z = 1/(1+sp.expit(-x))
  return z

def convert_vector_to_label(vector):
    vector = vector/(np.linalg.norm(vector) + 0.00001)
    angle = math.atan2(vector[1], vector[0])/math.pi*180
    if -22.5<= angle< 22.5:
        label = 0
    elif 22.5<= angle< 67.5:    
        label = 1   
    elif 67.5<= angle< 112.5:
        label = 2
    elif 112.5<= angle< 157.5:
        label = 3
    elif 157.5<= angle <= 180 or -180<= angle< -157.5:
        label = 4
    elif -157.5<= angle< -112.5:
        label = 5
    elif -112.5<= angle< -67.5:
        label = 6
    elif -67.5<= angle< -22.5:
        label = 7

    return np.array(int(label))

def convert_label_to_vector(label):
    match label:
        case 0:
            vector = np.array([1,0])
        case 1:
            vector = np.array([1,1])
        case 2:
            vector = np.array([0,1])
        case 3:
            vector = np.array([-1,1])
        case 4:
            vector = np.array([-1,0])
        case 5:
            vector = np.array([-1,-1])
        case 6:
            vector = np.array([0,-1])
        case 7:
            vector = np.array([1,-1])

    return vector/(np.linalg.norm(vector) + 0.000001)
    

def get_moving_avg(target_acq_time,window_size):
    acq_time_avg = np.zeros((len(target_acq_time)-window_size))
    for t in range(len(target_acq_time)-window_size):
        acq_time_avg[t] = np.mean(target_acq_time[t:t+window_size])
    return acq_time_avg


def save_results(model_name, rewards, time_to_target, path):

    with open(os.path.join(path, "test_data.txt"), "a") as f:
        f.write("model_name: " + str(model_name) + "\n")

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
        f.write("model_name: " + str(model_name) + "\n")

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
        
        
def check_shape(preds, labels):
	""" Checks that the shape of the predictions and labels are the same.
	"""
	if preds.shape != labels.shape:
		raise ValueError("preds and labels must have the same shape")

class r2():
    """ R2 Score of the model predictions.

    Currently implemented for 2D output only.
    """

    def __init__(self):
        """ Initalize metric state.

        Must hold memory of all labels seen so far.
        """
        self.x_sum_squares = 0.0
        self.y_sum_squares = 0.0
        
        self.x_labels = torch.tensor([])
        self.y_labels = torch.tensor([])

    def __call__(self, model, preds, data):
        """
        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: R2 Score.
        """
        check_shape(preds, data[1])
        self.x_sum_squares += torch.sum((data[1][:, 0] - preds[:, 0])**2).item()
        self.y_sum_squares += torch.sum((data[1][:, 1] - preds[:, 1])**2).item()
        self.x_labels = torch.cat((self.x_labels, data[1][:, 0]))
        self.y_labels = torch.cat((self.y_labels, data[1][:, 1]))

        return self.compute()

    def compute(self):
        """ Compute r2 score using accumulated data
        """
        x_denom = self.x_labels.var(correction=0)*len(self.x_labels)
        y_denom = self.y_labels.var(correction=0)*len(self.y_labels)

        x_r2 = 1 - (self.x_sum_squares/ x_denom)
        y_r2 = 1 - (self.y_sum_squares/ y_denom)

        r2 = (x_r2 + y_r2) / 2

        return r2.item()
    
def label_list_def(max_mag=0.5, min_mag=-0.5, label_number=8):
    range_total = max_mag - min_mag
    label_interval = range_total/label_number
    label_list = []
    current_value = min_mag
    for i in range(label_number-1):
        current_value += label_interval
        label_list.append(current_value)
    return label_list

def magnitude_to_label_class4(vector, max_mag=0.5, min_mag=-0.5, label_number=4):
    
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = vector.shape[0]
    label = torch.zeros((sequence_len, vector.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(vector[step, :]):
            if item <= label_list[0]:
                label[step,i] = 0
            elif label_list[0] <= item < label_list[1]:
                label[step,i] = 1
            elif label_list[1] <= item < label_list[2]:
                label[step,i] = 2
            elif label_list[2] <= item:
                label[step,i] = 3
                
    return label

def label_to_magnitude_class4(label, max_mag=0.5, min_mag=-0.5, label_number=4):
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = label.shape[0]
    vector = torch.zeros((sequence_len, label.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(label[step, :]):
            match item:
                case 0:
                    vector[step,i] = (label_list[0] - max_mag)/2
                case 1:
                    vector[step,i] = (label_list[0] + label_list[1])/2
                case 2:
                    vector[step,i] = (label_list[1] + label_list[2])/2
                case 3:
                    vector[step,i] = (label_list[2] + max_mag)/2
    return vector


def magnitude_to_label_class6(vector, max_mag=0.5, min_mag=-0.5, label_number=6):
    
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = vector.shape[0]
    label = torch.zeros((sequence_len, vector.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(vector[step, :]):
            if item <= label_list[0]:
                label[step,i] = 0
            elif label_list[0] <= item < label_list[1]:
                label[step,i] = 1
            elif label_list[1] <= item < label_list[2]:
                label[step,i] = 2
            elif label_list[2] <= item < label_list[3]:
                label[step,i] = 3
            elif label_list[3] <= item < label_list[4]:
                label[step,i] = 4
            elif label_list[4] <= item:
                label[step,i] = 5
                
    return label

def label_to_magnitude_class6(label, max_mag=0.5, min_mag=-0.5, label_number=6):
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = label.shape[0]
    vector = torch.zeros((sequence_len, label.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(label[step, :]):
            match item:
                case 0:
                    vector[step,i] = (label_list[0] - max_mag)/2
                case 1:
                    vector[step,i] = (label_list[0] + label_list[1])/2
                case 2:
                    vector[step,i] = (label_list[1] + label_list[2])/2
                case 3:
                    vector[step,i] = (label_list[2] + label_list[3])/2
                case 4:
                    vector[step,i] = (label_list[3] + label_list[4])/2
                case 5:
                    vector[step,i] = (label_list[4] + max_mag)/2
                    
    return vector


def magnitude_to_label_class8(vector, max_mag=0.5, min_mag=-0.5, label_number=8):
    
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = vector.shape[0]
    label = torch.zeros((sequence_len, vector.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(vector[step, :]):
            if item <= label_list[0]:
                label[step,i] = 0
            elif label_list[0] <= item < label_list[1]:
                label[step,i] = 1
            elif label_list[1] <= item < label_list[2]:
                label[step,i] = 2
            elif label_list[2] <= item < label_list[3]:
                label[step,i] = 3
            elif label_list[3] <= item < label_list[4]:
                label[step,i] = 4
            elif label_list[4] <= item < label_list[5]:
                label[step,i] = 5
            elif label_list[5] <= item < label_list[6]:
                label[step,i] = 6
            elif label_list[6] <= item:
                label[step,i] = 7
                
    return label

def label_to_magnitude_class8(label, max_mag=0.5, min_mag=-0.5, label_number=8):
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = label.shape[0]
    vector = torch.zeros((sequence_len, label.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(label[step, :]):
            match item:
                case 0:
                    vector[step,i] = (label_list[0] - max_mag)/2
                case 1:
                    vector[step,i] = (label_list[0] + label_list[1])/2
                case 2:
                    vector[step,i] = (label_list[1] + label_list[2])/2
                case 3:
                    vector[step,i] = (label_list[2] + label_list[3])/2
                case 4:
                    vector[step,i] = (label_list[3] + label_list[4])/2
                case 5:
                    vector[step,i] = (label_list[4] + label_list[5])/2
                case 6:
                    vector[step,i] = (label_list[5] + label_list[6])/2
                case 7:
                    vector[step,i] = (label_list[6] + max_mag)/2

                    
    return vector

def magnitude_to_label_class12(vector, max_mag=0.5, min_mag=-0.5, label_number=12):
    
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = vector.shape[0]
    label = torch.zeros((sequence_len, vector.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(vector[step, :]):
            if item <= label_list[0]:
                label[step,i] = 0
            elif label_list[0] <= item < label_list[1]:
                label[step,i] = 1
            elif label_list[1] <= item < label_list[2]:
                label[step,i] = 2
            elif label_list[2] <= item < label_list[3]:
                label[step,i] = 3
            elif label_list[3] <= item < label_list[4]:
                label[step,i] = 4
            elif label_list[4] <= item < label_list[5]:
                label[step,i] = 5
            elif label_list[5] <= item < label_list[6]:
                label[step,i] = 6
            elif label_list[6] <= item < label_list[7]:
                label[step,i] = 7
            elif label_list[7] <= item < label_list[8]:
                label[step,i] = 8
            elif label_list[8] <= item < label_list[9]:
                label[step,i] = 9
            elif label_list[9] <= item < label_list[10]:
                label[step,i] = 10
            elif label_list[10] <= item:
                label[step,i] = 11
                
    return label

def label_to_magnitude_class12(label, max_mag=0.5, min_mag=-0.5, label_number=12):
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = label.shape[0]
    vector = torch.zeros((sequence_len, label.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(label[step, :]):
            match item:
                case 0:
                    vector[step,i] = (label_list[0] - max_mag)/2
                case 1:
                    vector[step,i] = (label_list[0] + label_list[1])/2
                case 2:
                    vector[step,i] = (label_list[1] + label_list[2])/2
                case 3:
                    vector[step,i] = (label_list[2] + label_list[3])/2
                case 4:
                    vector[step,i] = (label_list[3] + label_list[4])/2
                case 5:
                    vector[step,i] = (label_list[4] + label_list[5])/2
                case 6:
                    vector[step,i] = (label_list[5] + label_list[6])/2
                case 7:
                    vector[step,i] = (label_list[6] + label_list[7])/2
                case 8:
                    vector[step,i] = (label_list[7] + label_list[8])/2
                case 9:
                    vector[step,i] = (label_list[8] + label_list[9])/2
                case 10:
                    vector[step,i] = (label_list[9] + label_list[10])/2
                case 11:
                    vector[step,i] = (label_list[10] + max_mag)/2
                    
    return vector


def magnitude_to_label_class16(vector, max_mag=0.5, min_mag=-0.5, label_number=16):
    
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = vector.shape[0]
    label = torch.zeros((sequence_len, vector.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(vector[step, :]):
            if item <= label_list[0]:
                label[step,i] = 0
            elif label_list[0] <= item < label_list[1]:
                label[step,i] = 1
            elif label_list[1] <= item < label_list[2]:
                label[step,i] = 2
            elif label_list[2] <= item < label_list[3]:
                label[step,i] = 3
            elif label_list[3] <= item < label_list[4]:
                label[step,i] = 4
            elif label_list[4] <= item < label_list[5]:
                label[step,i] = 5
            elif label_list[5] <= item < label_list[6]:
                label[step,i] = 6
            elif label_list[6] <= item < label_list[7]:
                label[step,i] = 7
            elif label_list[7] <= item < label_list[8]:
                label[step,i] = 8
            elif label_list[8] <= item < label_list[9]:
                label[step,i] = 9
            elif label_list[9] <= item < label_list[10]:
                label[step,i] = 10
            elif label_list[10] <= item < label_list[11]:
                label[step,i] = 11
            elif label_list[11] <= item < label_list[12]:
                label[step,i] = 12
            elif label_list[12] <= item < label_list[13]:
                label[step,i] = 13
            elif label_list[13] <= item < label_list[14]:   
                label[step,i] = 14
            elif label_list[14] <= item:
                label[step,i] = 15

    return label

def label_to_magnitude_class16(label, max_mag=0.5, min_mag=-0.5, label_number=16):
    label_list = label_list_def(max_mag=max_mag, min_mag=min_mag, label_number=label_number)
    sequence_len = label.shape[0]
    vector = torch.zeros((sequence_len, label.shape[1]))
    
    for step in range(sequence_len):
        for i, item in enumerate(label[step, :]):
            match item:
                case 0:
                    vector[step,i] = (label_list[0] - max_mag)/2
                case 1:
                    vector[step,i] = (label_list[0] + label_list[1])/2
                case 2:
                    vector[step,i] = (label_list[1] + label_list[2])/2
                case 3:
                    vector[step,i] = (label_list[2] + label_list[3])/2
                case 4:
                    vector[step,i] = (label_list[3] + label_list[4])/2
                case 5:
                    vector[step,i] = (label_list[4] + label_list[5])/2
                case 6:
                    vector[step,i] = (label_list[5] + label_list[6])/2
                case 7:
                    vector[step,i] = (label_list[6] + label_list[7])/2
                case 8:
                    vector[step,i] = (label_list[7] + label_list[8])/2
                case 9:
                    vector[step,i] = (label_list[8] + label_list[9])/2
                case 10:
                    vector[step,i] = (label_list[9] + label_list[10])/2
                case 11:
                    vector[step,i] = (label_list[10] + label_list[11])/2
                case 12:
                    vector[step,i] = (label_list[11] + label_list[12])/2
                case 13:
                    vector[step,i] = (label_list[12] + label_list[13])/2
                case 14:
                    vector[step,i] = (label_list[13] + label_list[14])/2
                case 15:
                    vector[step,i] = (label_list[14] + max_mag)/2

                    
    return vector
