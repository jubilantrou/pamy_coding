import pickle5 as pickle
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from CNN import CNN
import os
import time
# %% add the limits
"""
max_pressure_ago = [22000, 25000, 22000, 22000]
max_pressure_ant = [22000, 23000, 22000, 22000]

min_pressure_ago = [13000, 13500, 10000, 8000]
min_pressure_ant = [13000, 14500, 10000, 8000]
"""

"""
anchor = [17500 18500 16000 15000]
"""
limit_max = np.array([4500, 6500, 6000, 7000])
limit_min = np.array([-4500, -5000, -6000, -7000])
limit_mean = ( limit_max + limit_min ) / 2
limit_err = limit_max - limit_mean
# %%
def LCNNPrediction( t_stamp, desired_path_in, path=None ):
# %% 
    desired_path = desired_path_in.copy()
    path_of_file = "/home/hao/Desktop/Hao/cnn_model"
    pi = math.pi
    # half length of inputs
    q_input = 100
    # half length of outputs
    q_output = 5
    # position of middle point (main point)
    mid_position = q_output
    # two channels: desired path and initial feedforward
    channel_in = 2
    channel_out = 1
    width_in = 3
    width_out = 1
    # length of inputs
    length_of_input = 2 * q_input + 1
    # length of labels
    length_of_output = 2 * q_output + 1
# %% read mean and std from file
    mean = np.array([  0.66712741, -11.23271696, -19.90662067])
    std = np.array([26.69764846, 16.37475003, 18.13647442])
# %% normalization    
    desired_path = desired_path * 180 / pi
    for dof in range( 3 ):
        desired_path[dof, :] = (desired_path[dof, :] - mean[dof]) / std[dof] 
# %% read parameter tensor A from file
    path_of_LModel = path_of_file + '/' + 'A_list'
    file = open(path_of_LModel, 'rb')
    A_list = pickle.load(file)
    file.close()
# %% generate data
    dataset = []
    # dimension = length * 3
    desired_path = desired_path.T
    l = desired_path.shape[0]

    for i in range( l ):
        y_temp = np.array([])
        # loop for inputs
        for j in range( i-q_input, i+q_input+1):
            if j < 0:
                y_temp = np.append(y_temp, desired_path[0, :] )       
            elif j > l-1:
                y_temp = np.append(y_temp, desired_path[-1, :] )
            else:
                y_temp = np.append(y_temp, desired_path[j, :] )
       
        # dimension 1 * (3 * length + 1)
        y_temp = torch.tensor( y_temp ).view(1, -1)
        bias = torch.tensor(1).view(1, -1)
        data_temp = torch.cat( [y_temp, bias], dim=1 )
    
        dataset.append( data_temp )
# %% generate linear model
    linear_model = np.array([])

    for i in range( len( dataset ) ):

        for dof in range( 3 ):
            
            pred = torch.matmul( dataset[i], A_list[dof] )
            pred = pred.detach().numpy()
            pred = ( np.asscalar( pred ) - limit_mean[dof] ) / limit_err[dof] 

            if pred > 1:
                pred = 1
            if pred < -1:
                pred = -1

            linear_model = np.append(linear_model, pred)

    linear_model = linear_model.reshape(-1, 3)  
# %% generate the data for CNN
    dataset = []

    l = desired_path.shape[0]

    for i in range( l ):
        y_temp = np.array([])
        u_ini_temp = np.array([])
        # loop for inputs
        for j in range( i-q_input, i+q_input+1):
            if j < 0:
                # y_temp = np.append(y_temp, desired_path[0, :] )
                # u_ini_temp = np.append(u_ini_temp, linear_model[0, :] )   
                y_temp = np.append(y_temp, np.zeros(3) )
                u_ini_temp = np.append(u_ini_temp, np.zeros(3) )         
            elif j > l-1:
                # y_temp = np.append(y_temp, desired_path[-1, :] )
                # u_ini_temp = np.append(u_ini_temp, linear_model[-1, :] )  
                y_temp = np.append(y_temp, np.zeros(3) )
                u_ini_temp = np.append(u_ini_temp, np.zeros(3) )  
            else:
                y_temp = np.append(y_temp, desired_path[j, :] )
                u_ini_temp = np.append(u_ini_temp, linear_model[j, :] )  
            # dimension = batch_size * channels * height * width 
        y_temp = torch.tensor( y_temp ).view(1, 1, -1, width_in)
        u_ini_temp = torch.tensor( u_ini_temp ).view(1, 1, -1, width_in)

        data_temp = torch.cat( [y_temp, u_ini_temp], dim=1 )

        dataset.append( data_temp )
# %% import CNN model from file
    cnn_model_list = []
    map_location = torch.device('cpu')
    for dof in range( 3 ):
        path_of_CModel = path_of_file + '/' + 'model_dof_' + str(dof)
        path_of_CModel_model = path_of_CModel + '/' + 'model'
        path_of_CModel_parameter =  path_of_CModel + '/' + 'model_parameter'
        cnn_model = torch.load( path_of_CModel_model, map_location='cpu' )
        cnn_model.load_state_dict( torch.load(path_of_CModel_parameter, map_location='cpu') )
        cnn_model_list.append( cnn_model )
# %% prediction
    ff = np.array([])
    for i in range( len(dataset) ):

        for dof in range( 3 ):

            preds = cnn_model_list[dof]( dataset[i].float() )
            preds = preds.squeeze().cpu()
            preds = preds.detach().numpy().reshape(1, - 1, 1)

            ff = np.append( ff, preds[0, mid_position, 0] )

    ff = ff.reshape(-1, 3)
    ff = ff.T
# %% denormalization
    for dof in range( 3 ):
        ff[dof, :] = ( ff[dof, :] * limit_err[dof] ) + limit_mean[dof]

    ff_dof_3 = np.zeros( len( t_stamp ) ).reshape(1, -1)

    ff = np.vstack((ff, ff_dof_3))  
# %% save data
    path_of_training = path + '/' + 'training_data'
    file = open(path_of_training, 'wb')
    pickle.dump(desired_path, file, -1) # time stamp for x-axis
    pickle.dump(linear_model, file, -1)
    file.close()

    return ff