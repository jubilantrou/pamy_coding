import torch
import numpy as np
import math
from Trainable_blocks import CNN
import numba as nb
import time
# %% add the limits
"""
max_pressure_ago = [22000, 25000, 22000, 22000]
max_pressure_ant = [22000, 23000, 22000, 22000]

min_pressure_ago = [13000, 13500, 10000, 8000]
min_pressure_ant = [13000, 14500, 10000, 8000]

anchor = [17500 18500 16000 15000]
"""
limit_max = np.array([4500, 6500, 6000, 7000])
limit_min = np.array([-4500, -5000, -6000, -7000])
limit_mean = ( limit_max + limit_min ) / 2
limit_err = limit_max - limit_mean
# %% global constant
pi = math.pi
# half length of inputs
q_input = 100
# half length of outputs
q_output = 5
# position of middle point (main point)
mid_position = q_output

mean = np.array([0.66712741, -11.23271696, -19.90662067])
std = np.array([26.69764846, 16.37475003, 18.13647442])

def cnn( model , data ):
    pred = model( data ).squeeze().detach().numpy()
    return pred[mid_position]

@nb.jit(nopython=True)
def quick_linear(up_limit, down_limit, A_list, A_bias, y_temp):
    linear_model = np.zeros( (up_limit - down_limit, 3) )
    for i in range( down_limit, up_limit):
        linear_model[i - down_limit, :] = (np.dot( y_temp[i:i+2*q_input+1, :].ravel(), A_list) + A_bias - limit_mean[0:3] ) / limit_err[0:3] 
    return linear_model

def LCNN(k, desired_path_in, A_list, A_bias, cnn_model_list ):
    desired_path = desired_path_in.copy()
    desired_path = (desired_path.T * 180 / pi - mean) / std 
# %% generate data for linear prediction
    y_temp = np.vstack((np.tile( desired_path[0, :], q_input).reshape(q_input, 3), desired_path, np.tile( desired_path[-1, :], q_input).reshape(q_input, 3)))
# %% linear prediction
    down_limit = 0 if k-q_input < 0 else k-q_input
    up_limit = desired_path.shape[0] if k+q_input+1 >= desired_path.shape[0] else k+q_input+1
    linear_model = quick_linear(up_limit, down_limit, A_list, A_bias, y_temp)
# %% generate the data for CNN
    left_compensate = abs(k-q_input) if k-q_input<0 else 0
    right_compensate = k+q_input+1-desired_path.shape[0] if k+q_input+1 >= desired_path.shape[0] else 0

    if left_compensate > 0 and right_compensate > 0:
        y_temp = np.vstack( (np.zeros((left_compensate, 3)), desired_path[down_limit:up_limit, :], np.zeros((right_compensate, 3)) ) )
        linear_temp = np.vstack( (np.zeros((left_compensate, 3)), linear_model, np.zeros((right_compensate, 3)) ) )
    
    elif left_compensate > 0 and right_compensate == 0:
        y_temp = np.vstack( (np.zeros((left_compensate, 3)), desired_path[down_limit:up_limit, :] ) )
        linear_temp = np.vstack( (np.zeros((left_compensate, 3)), linear_model ) )
    
    elif left_compensate == 0 and right_compensate > 0:
        y_temp = np.vstack( ( desired_path[down_limit:up_limit, :], np.zeros((right_compensate, 3)) ) )
        linear_temp = np.vstack( (linear_model, np.zeros((right_compensate, 3)) ) ) 
    
    elif left_compensate == 0 and right_compensate == 0:
        y_temp = np.copy( desired_path[down_limit:up_limit, :] )
        linear_temp = np.copy( linear_model )

    data = torch.cat( [torch.tensor( y_temp ).view(1, 1, -1, 3), torch.tensor( linear_temp ).view(1, 1, -1, 3)], dim=1 )
# %% prediction & denormalization
    ff = np.zeros(4)
    
    for i in range(3):
        ff[i] = cnn(cnn_model_list[i], data.float()) * limit_err[i] + limit_mean[i]

    return ff