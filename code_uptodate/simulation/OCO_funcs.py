'''
This script is used to define the related functions 
for the OCO training procedure.
'''
import PAMY_CONFIG
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import random
import o80_pam
import torch
from CNN import CNN
import torch.nn as nn
import time
from RealRobotGeometry import RobotGeometry
import torch.nn as nn
import o80
import wandb
from pyinstrument import Profiler

def fix_seed(seed):
    '''
    to ensure the reproducibility
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_random():
    '''
    to randomly generate the target position and the T_go
    '''
    theta = np.zeros(3)
    theta[0] = random.choice([random.randrange(-700, -250)/10, random.randrange(250, 700)/10])
    theta[1] = random.randrange(150, 750)/10
    theta[2] = random.randrange(150, 750)/10
    t        = random.randrange(90, 100)/100
    theta    = theta * math.pi/180
    return (t, theta)

def get_compensated_data(data, h_l, h_r, option=None):
    '''
    to pad the reference trajectory
    '''
    I_left = np.tile(data[:, 0].reshape(-1, 1), (1, h_l))
    I_right = np.tile(data[:, -1].reshape(-1, 1), (1, h_r))
    if option=='only_left':
        aug_data = np.hstack((I_left, data))
    else:
        aug_data = np.hstack((I_left, data, I_right))
    return aug_data  

def get_datapoint(y, h_l, h_r, ds, device, sub_traj=None, ref=None):
    '''
    to get the datapoint from the reference trajectory
    '''
    l = y.shape[1]
    aug_y = get_compensated_data(y, h_l, h_r)
    datapoint = []

    for k in range(l):
        if sub_traj is None:
            y_temp = np.concatenate((aug_y[0, k:k+(h_l+h_r)+1:ds].reshape(1,-1), aug_y[1, k:k+(h_l+h_r)+1:ds].reshape(1,-1), aug_y[2, k:k+(h_l+h_r)+1:ds].reshape(1,-1)), axis=0)
        else:
            choice = 0
            for ele in ref:
                if k>ele:
                    choice += 1
                else:
                    break
            if choice<len(ref):
                y_temp = np.concatenate((sub_traj[choice][0, k:k+(h_l+h_r)+1:ds].reshape(1,-1), sub_traj[choice][1, k:k+(h_l+h_r)+1:ds].reshape(1,-1), sub_traj[choice][2, k:k+(h_l+h_r)+1:ds].reshape(1,-1)), axis=0)
            else:
                y_temp = np.concatenate((aug_y[0, k:k+(h_l+h_r)+1:ds].reshape(1,-1), aug_y[1, k:k+(h_l+h_r)+1:ds].reshape(1,-1), aug_y[2, k:k+(h_l+h_r)+1:ds].reshape(1,-1)), axis=0)            
        # data: (channel x height x width)
        datapoint.append(torch.tensor(y_temp, dtype=float).view(-1).to(device))
    
    return datapoint

def get_dataset(datapoint, batch_size):
    '''
    to construct the dataset from the datapoint
    '''
    l = len(datapoint)
    dataset = []
    idx = 0
    while idx + batch_size - 1 < l:
        data_ = datapoint[idx:idx+batch_size]
        batch = torch.stack(data_)
        # elements in dataset: batchsize x (channel x height x width)
        dataset.append(batch)
        idx += batch_size

    return dataset

def get_prediction(datapoint, cnn_list, y):
    '''
    to get the trainable part of the contorl inputs
    '''
    dataset = get_dataset(datapoint=datapoint, batch_size=y.shape[1])
    u = np.zeros(y.shape)
    for dof,cnn in enumerate(cnn_list):
        if cnn is None:
            continue
        else:
            cnn.eval()
            try:
                u[dof, :] = cnn(dataset[0]).cpu().detach().numpy().flatten()
            except:
                u[dof, :] = cnn(dataset[0].float()).cpu().detach().numpy().flatten()
    return u

def wandb_plot(i_iter, frequency, t_stamp, ff, fb, y, theta_, t_stamp_list, theta_list, T_go_list, p_int_record):
    plots = []
    if (i_iter+1)%frequency==0:
        legend_position = 'best'
        fig1 = plt.figure(figsize=(18, 18))

        ax1_position0 = fig1.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_0')
        line = []
        line_temp, = ax1_position0.plot(t_stamp, ff[0, :], linewidth=2, label=r'uff_Dof0')
        line.append( line_temp )
        line_temp, = ax1_position0.plot(t_stamp, fb[0, :], linewidth=2, label=r'ufb_Dof0')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
            
        ax1_position1 = fig1.add_subplot(312)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_1')
        line = []
        line_temp, = ax1_position1.plot(t_stamp, ff[1, :], linewidth=2, label=r'uff_Dof1')
        line.append( line_temp )
        line_temp, = ax1_position1.plot(t_stamp, fb[1, :], linewidth=2, label=r'ufb_Dof1')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
        
        ax1_position2 = fig1.add_subplot(313)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_2')
        line = []
        line_temp, = ax1_position2.plot(t_stamp, ff[2, :], linewidth=2, label=r'uff_Dof2')
        line.append( line_temp )
        line_temp, = ax1_position2.plot(t_stamp, fb[2, :], linewidth=2, label=r'ufb_Dof2')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Pressure Input'+' Iter '+str(i_iter+1))
        plots.append(wandb.Image(plt, caption="matplotlib image"))                

        fig = plt.figure(figsize=(18, 18))

        ax_position0 = fig.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position of Dof_0 in degree')
        line = []
        line_temp, = ax_position0.plot(t_stamp, y[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof0_out')
        line.append( line_temp )
        line_temp, = ax_position0.plot(t_stamp, theta_[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof0_des')
        line.append( line_temp )
        for j in range(len(theta_list)):
            line_temp, = ax_position0.plot(t_stamp_list[j], theta_list[j][0, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof0_traj_candidate_'+str(j+1))
            line.append( line_temp )
            line_temp, = ax_position0.plot(T_go_list[j], p_int_record[j][0] * 180 / math.pi, 'o', label='target_'+str(j+1))
            line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
            
        ax_position1 = fig.add_subplot(312)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position of Dof_1 in degree')
        line = []
        line_temp, = ax_position1.plot(t_stamp, y[1, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof1_out')
        line.append( line_temp )
        line_temp, = ax_position1.plot(t_stamp, theta_[1, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof1_des')
        line.append( line_temp )
        for j in range(len(theta_list)):
            line_temp, = ax_position1.plot(t_stamp_list[j], theta_list[j][1, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof1_traj_candidate_'+str(j+1))
            line.append( line_temp )
            line_temp, = ax_position1.plot(T_go_list[j], p_int_record[j][1] * 180 / math.pi, 'o', label='target_'+str(j+1))
            line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
        
        ax_position2 = fig.add_subplot(313)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position of Dof_2 in degree')
        line = []
        line_temp, = ax_position2.plot(t_stamp, y[2, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof2_out')
        line.append( line_temp )
        line_temp, = ax_position2.plot(t_stamp, theta_[2, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof2_des')
        line.append( line_temp )
        for j in range(len(theta_list)):
            line_temp, = ax_position2.plot(t_stamp_list[j], theta_list[j][2, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof2_traj_candidate_'+str(j+1))
            line.append( line_temp )
            line_temp, = ax_position2.plot(T_go_list[j], p_int_record[j][2] * 180 / math.pi, 'o', label='target_'+str(j+1))
            line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Joint Space Trajectory Tracking Performance'+' Iter '+str(i_iter+1))
        plots.append(wandb.Image(plt, caption="matplotlib image"))                
        
        wandb.log({'related plots': plots})
        plt.close()

def get_grads_list(dataset, cnn_list):
    '''
    Nt x nff
    '''
    X_list = []

    for cnn in cnn_list:
        if cnn is None:
            X_list.append(None)
            continue

        cnn.train()
        for param in cnn.parameters():
            if param.grad is None:
                break
            param.grad.zero_()

        flag = True
        for data in dataset:
            grad = []
            try:
                cnn(data.float()).mean().backward()
            except:
                cnn(data).mean().backward()                       
            for param in cnn.parameters():
                grad.append(torch.clone(param.grad.cpu().view(-1)))
                param.grad.zero_()
                
            grads = torch.cat(grad)            
            grads_ = np.copy(grads.reshape(1, -1)) if flag else np.concatenate((grads_, grads.reshape(1, -1)), axis=0)
            flag = False if flag else False

        X_list.append(grads_)

    return X_list

def set_parameters(W_list, cnn_list, idx_list, shape_list, device):
    for dof in range(len(cnn_list)):
        cnn = cnn_list[dof]
        if cnn is None:
            continue
        W = W_list[dof]
        i = 0
        for param in cnn.parameters():
            idx_1 = idx_list[i]
            idx_2 = idx_list[i+1]
            W_ = torch.tensor(W[idx_1:idx_2]).view(shape_list[i])
            param.data = W_.to(device)
            i += 1
    return cnn_list

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path) 

# def get_step_size(nr, dof, step_size_version='constant'):
#     factor = [0.1, 0.1, 0.1]
#     constant_list = np.copy(paras.lr_list)
#     if step_size_version == 'constant':
#         step_size = constant_list[dof]
#     elif step_size_version == 'sqrt':
#         step_size = factor[dof]/(2+np.sqrt(nr))
#     return step_size 
