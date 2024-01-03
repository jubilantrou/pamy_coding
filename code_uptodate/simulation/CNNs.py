'''
This script is used to generate cnn models for 
all degrees of freedom. It includes 2-channel model
and 3-channel model
'''
import torch
import numpy as np
import pickle5 as pickle
import math
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_norm(y):
    limit_joint = np.array([90, 60, 60]) * math.pi / 180
    for dof in range(3):
        y[dof, :] = y[dof, :] /limit_joint[dof]
    return y

def get_norm_ff(ff):
    limit_pressure = np.array([4500, 6500, 6000])
    for dof in range(3):
        ff[dof, :] = ff[dof, :] / limit_pressure[dof]
    return ff


def get_denorm(ff):
    limit_pressure = np.array([4500, 6500, 6000])
    for dof in range(3):
        ff[dof, :] = ff[dof, :] * limit_pressure[dof]
    return ff

def get_diff(y):
    x = np.zeros(y.shape)
    for i in range(1, y.shape[1]):
        x[:, i] = (y[:, i] - y[:, i-1]) / 0.01
    return x

def get_channel_data(dof_list=[0,1,2], h_left=10, h_right=10, data=None):
    outputs = []
    l = data.shape[1]

    for i in range( l ):
        y_temp = np.array([])
        # loop for inputs
        for j in range( i-h_left, i+h_right+1):
            if j < 0:
                y_temp = np.append(y_temp, np.zeros(len(dof_list)))
            elif j > l-1:
                y_temp = np.append(y_temp, np.zeros(len(dof_list)))
            else:
                y_temp = np.append(y_temp, data[dof_list, j])
        y_temp = torch.tensor(y_temp.reshape(-1, len(dof_list))).view(1, 1, -1, len(dof_list))
        outputs.append( y_temp )

    return outputs

def get_all_channel(data, channel):
    outputs = []
    for i in range(len(data[0])):
        if channel == 2:
            outputs.append(torch.cat((data[0][i], data[1][i]), dim=1))
        elif channel  == 3:
            outputs.append(torch.cat((data[0][i], data[1][i], data[2][i]), dim=1))
    return outputs

def get_data(dof_list=[0,1,2], h_in_left=20, h_in_right=20, input_data=None):
    channel = len(input_data)  # how mang channels do we use
    channel_data = []
    for i in range(channel):
        channel_data.append(get_channel_data(dof_list=dof_list, h_left=h_in_left, 
                                             h_right=h_in_right, data=input_data[i]))
    if channel > 1:
        dataset = get_all_channel(channel_data, channel)
    else:
        dataset = channel_data[0]

    return dataset

def get_cnn_ff(model, y):

    cnn_model_list = model[0]

    if len(model) == 1:
        number_channel = 3
    elif len(model) == 2:
        number_channel = 2

    ff = np.zeros(y.shape)
    y = get_norm(y)
    vel = get_diff(y)
    acc = get_diff(vel)

    if number_channel == 3:
        data = get_data(input_data=(y, vel, acc))
    elif number_channel == 2:
        lienar_prediction = get_norm_ff(get_linear_ff(model[1], y))
        data = get_data(input_data=(y, lienar_prediction))

    cnn_input = torch.cat(data, dim=0)

    for dof in range(3):
        cnn_model = cnn_model_list[dof]
        ff[dof, :] = cnn_model(cnn_input.float()).cpu().detach().numpy().flatten()

    ff = get_denorm(ff)    
    return ff

def get_ilc_ff(index, path=None):

    if path == None:
        path = '/home/hao/Desktop/HitBall/data/Result_ILC'
    f = open(path+'/'+'ILC_Ball_'+str(index), 'rb')
    t_stamp = pickle.load(f)
    t_list = pickle.load(f)
    angle_initial = pickle.load(f)
    y_history = pickle.load(f)
    repeated = pickle.load(f)
    y_pid = pickle.load(f)
    ff_history = pickle.load(f)
    fb_history = pickle.load(f)
    ago_history = pickle.load(f)
    ant_history = pickle.load(f)
    d_history = pickle.load(f)
    P_history = pickle.load(f)
    d_lifted_history = pickle.load(f)
    P_lifted_history = pickle.load(f)
    f.close()

    ff = ff_history[-1]
    y = y_history[0] + angle_initial.reshape(-1, 1)

    return (y, ff)

def get_linear_data(dof_list=[0,1,2], h_left=10, h_right=10, data=None):
    outputs = []
    l = data.shape[1]
    bias = torch.tensor(1).view(1, -1)
    for i in range( l ):
        y_temp = np.array([])
        # loop for inputs
        for j in range( i-h_left, i+h_right+1):
            if j < 0:
                y_temp = np.append(y_temp, np.zeros(len(dof_list)))
            elif j > l-1:
                y_temp = np.append(y_temp, np.zeros(len(dof_list)))
            else:
                y_temp = np.append(y_temp, data[dof_list, j])
        y_temp = torch.tensor(y_temp.reshape(-1, len(dof_list)).T.flatten()).view(1, -1)
        y_temp = torch.cat([y_temp, bias], dim=1)
        outputs.append(y_temp)
    return outputs

def get_linear_ff(model, y):
    ff = np.zeros(y.shape)
    y = get_norm(y)
    data = torch.cat(get_linear_data(data=y, h_left=100, h_right=100), dim=0).squeeze()
    for i in range(len(model)):
        ff[i, :] = torch.matmul(data, model[i]).numpy().flatten()
    ff = get_denorm(ff)
    return ff

def get_model_3_ch(root_path=None, device='cpu'):
    if root_path == None:
        root_path = '/home/hao/Desktop/HitBall/CNN/model_3_ch'
    model_list = []
    for i in range(3):
        path  = root_path + '/' + 'dof_' + str(i)
        model = torch.load(path+'/'+'model', map_location=device)
        model.load_state_dict(torch.load(path+'/'+'model_parameter', map_location=device))
        model_list.append(model)
    return model_list

def get_model_2_ch(root_path=None, device='cpu'):
    if root_path == None:
        root_path = '/home/hao/Desktop/HitBall/CNN/model_2_ch'
    model_list = []
    for i in range(3):
        path  = root_path + '/' + 'dof_' + str(i)
        model = torch.load(path+'/'+'model', map_location=device)
        model.load_state_dict(torch.load(path+'/'+'model_parameter', map_location=device))
        model_list.append(model)
    f = open(root_path+'/'+'linear_model', 'rb')
    linear_model = pickle.load(f)
    f.close()
    return (model_list, linear_model)

