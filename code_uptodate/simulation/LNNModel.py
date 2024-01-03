import pickle5 as pickle
import torch
import torch.nn as nn
from torch.nn.functional import dropout
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time
# %%
def LNNModel(q=5, k=0.7, scr=1000, desired_path_list_in=None,
             ff_list_in=None, t_stamp_list_in=None, 
             points_in=None):
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
    desired_path_list = desired_path_list_in.copy()
    ff_list = ff_list_in.copy()
    t_stamp_list = t_stamp_list_in.copy()
    points = points_in.copy()

    # result
    out_dof_list = [0, 1, 2]
    # input dof
    in_dof_list = [0, 1, 2]
# %% cheak
    # for i in range( len(desired_path_list) ):
    #     t_stamp = t_stamp_list[i]
    #     line = []
    #     plt.figure( r'Prediction for dof: {}'.format(dof) )
    #     plt.xlabel(r'time t in s')
    #     plt.ylabel(r'Feedforward control')
    #     line_temp, = plt.plot(t_stamp, desired_path_list[i][0, :], linewidth=1, label=r'0')
    #     line.append( line_temp )
    #     line_temp, = plt.plot(t_stamp, desired_path_list[i][1, :], linewidth=1, label=r'1')
    #     line.append( line_temp )
    #     line_temp, = plt.plot(t_stamp, desired_path_list[i][2, :], linewidth=1, label=r'2')
    #     line.append( line_temp )
    #     plt.show()
# %% set some constants
    # total number of training paths
    number_path = len( desired_path_list )
    # half length of inputs M
    q_left = q
    # half length of outputs N
    q_right = q
    # length of inputs
    length_of_input = q_left + q_right + 1
    # how big part is used to train CNN model
    # %% generate the dataset and labelset
    dataset = []
    labelset = []

    for index in range( number_path ):
        # the dimensions should be 4 * n
        desired_path = desired_path_list[index].T
        ff = ff_list[index].T
        
        l = desired_path.shape[0]

        for i in range( l ):
            y_temp = np.array([])
            u_temp = np.array([])
            # loop for inputs
            for j in range( i-q_left, i+q_right+1):
                if j < 0:
                    y_temp = np.append(y_temp, desired_path[0, :] )
                elif j > l-1:
                    y_temp = np.append(y_temp, desired_path[-1, :] )
                else:
                    y_temp = np.append(y_temp, desired_path[j, :] )
            # loop for outputs
            for j in range( i, i+1):
                if j < 0:
                    u_temp = np.append(u_temp, ff[0, :] )
                elif j > l-1:
                    u_temp = np.append(u_temp, ff[-1, :] )
                else:
                    u_temp = np.append(u_temp, ff[j, :] )

            # add the bias
            # dimension 1 * (3 * length + 1)
            y_temp = torch.tensor( y_temp ).view(1, -1)
            bias = torch.tensor(1).view(1, -1)
            # dimension 1 * 3
            u_temp = torch.tensor( u_temp ).view(-1, 3)

            data_temp = torch.cat( [y_temp, bias], dim=1 )
  
            dataset.append( data_temp )
            # 1
            labelset.append( u_temp )

    path_total = len( desired_path_list )
    path_to_train = round( k * path_total )

    num_point = round(sum( points ))
    num_train = round(sum( points[0:path_to_train] ))
    num_val = num_point - num_train
    arr = np.arange( num_point )

    val_data = []
    val_label = []

    for i in range( num_train ):
        if i == 0:
            train_data = dataset[i]
            train_label = labelset[i]
        else:
            # p * (3 * length + 1)
            train_data = torch.cat((train_data, dataset[i]), 0)
            train_label = torch.cat((train_label, labelset[i]), 0)
# %%
    # train_data = train_data.view( -1, channels * dof * length_of_input + 1 )
    theta_list = []
    for dof in range( len( out_dof_list ) ):
        theta = torch.matmul( torch.pinverse( train_data ), train_label[:, dof] )
        theta_list.append( theta )
    
    theta_to_save = '/home/hao/Desktop/pamy/LCNNModel/A_list'
    file = open(theta_to_save, 'wb')
    pickle.dump(theta_list, file, -1) # time stamp for x-axis
    file.close()
    # %%
    linear_model_list = []

    Location = 'lower right'

    i_left = 0
    i_right = 0

    for i in range( path_to_train ):

        i_right += round( points[i] )
        t_stamp = t_stamp_list[i]
        linear_model_dof = np.array([])

        for dof in range( len( out_dof_list ) ):
            
            u_model = np.array([])
            u_eval = np.array([])
            for index in range(i_left, i_right):

                pred = torch.matmul( dataset[index], theta_list[dof] )
                pred = pred.detach().numpy()
                pred = ( np.asscalar( pred ) * scr - limit_mean[dof] ) / limit_err[dof] 

                if pred > 1:
                    pred = 1
                if pred < -1:
                    pred = -1

                u_model = np.append(u_model, pred)
                u_eval = np.append(u_eval, ( np.asscalar( labelset[index][:, dof] ) * scr - limit_mean[dof] ) / limit_err[dof] )

            linear_model_dof = np.append( linear_model_dof, u_model )

            # line = []

            # plt.figure( r'Prediction dof {}'.format(dof) )
            # plt.xlabel(r'time t in s')
            # plt.ylabel(r'Feedforward control')
            # line_temp, = plt.plot(t_stamp, u_model, linewidth=1, label=r'Training')
            # line.append( line_temp )
            # line_temp, = plt.plot(t_stamp, u_eval, linewidth=1.5, linestyle='--', label=r'Desired')
            # line.append( line_temp )
            # plt.legend(handles = line, loc=Location, shadow=True)
            # plt.show()

        linear_model_dof = linear_model_dof.reshape(len(out_dof_list), -1)
        linear_model_list.append( linear_model_dof )

        i_left = i_right
    # %%
    for i in range(path_to_train, path_total):
        
        i_right += round( points[i] )

        
        t_stamp = t_stamp_list[i]
        linear_model_dof = np.array([])
        for dof in range( len( out_dof_list ) ):
            
            u_model = np.array([])
            u_eval = np.array([])
            for index in range(i_left, i_right):

                pred = torch.matmul( dataset[index], theta_list[dof] )
                pred = pred.detach().numpy()
                pred = ( np.asscalar( pred ) * scr - limit_mean[dof] ) / limit_err[dof] 

                if pred > 1:
                    pred = 1
                if pred < -1:
                    pred = -1

                u_model = np.append(u_model, pred)
                u_eval = np.append(u_eval, ( np.asscalar( labelset[index][:, dof] ) * scr - limit_mean[dof] ) / limit_err[dof] )

            linear_model_dof = np.append( linear_model_dof, u_model )

            # line = []

            # plt.figure( r'Prediction dof {}'.format(dof) )
            # plt.xlabel(r'time t in s')
            # plt.ylabel(r'Feedforward control')
            # line_temp, = plt.plot(t_stamp, u_model, linewidth=1, label=r'Prediction')
            # line.append( line_temp )
            # line_temp, = plt.plot(t_stamp, u_eval, linewidth=1.5, linestyle='--', label=r'Desired')
            # line.append( line_temp )
            # plt.legend(handles = line, loc=Location, shadow=True)

            # plt.show()
        
        linear_model_dof = linear_model_dof.reshape(len(out_dof_list), -1)
        linear_model_list.append( linear_model_dof )

        i_left = i_right
    
    return (linear_model_list)
