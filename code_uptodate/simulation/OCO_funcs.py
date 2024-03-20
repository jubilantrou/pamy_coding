'''
This script is used to define the related functions 
for the OCO training procedure.
'''
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import random
import torch
import wandb

def fix_seed(seed):
    '''
    to ensure the reproducibility
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_random():
    '''
    to randomly generate the mimic interception time point and the mimic interception position

    Returns:
        t: the mimic interception time point, in seconds
        theta: the mimic interception position, in rads and in the joint space
    '''
    t = random.randrange(90, 100)/100
    theta = np.zeros(3)
    theta[0] = random.choice([random.randrange(-900, -450)/10, random.randrange(450, 900)/10])
    theta[1] = random.randrange(150, 750)/10
    theta[2] = random.randrange(150, 750)/10
    theta = theta * math.pi/180
    return (t, theta)

def get_compensated_data(data, h_l, h_r):
    '''
    to pad the reference trajectory using replicate padding for ease of sliding a window along it

    Args:
        data: the reference trajectory to be padded
        h_l: the window length to the left of the centered time point
        h_r: the window length to the right of the centered time point
        (thus, the whole window length will be h_l+h_r+1)
    Returns:
        aug_data: the padded reference trajectory
    '''
    I_left = np.tile(data[:, 0].reshape(-1, 1), (1, h_l))
    I_right = np.tile(data[:, -1].reshape(-1, 1), (1, h_r))
    aug_data = np.hstack((I_left, data, I_right))
    return aug_data

def get_datapoints(aug_data, window_size, ds, device, nn_type):
    '''
    to extract datapoints of each time step from the padded reference trajectory

    Args:
        aug_data: the padded reference trajectory
        window_size: the whole length of the window
        ds: the stride for down sampling inside the window
        device: where to send datapoints for training later
        nn_type: the type of the neural network we used for training, which determines the dimension of datapoints
    Returns:
        datapoints: extracted datapoints
    '''
    datapoints = []

    l = aug_data.shape[1] - window_size + 1
    for k in range(l):
        datapoint = aug_data[0:3, k:k+window_size:ds].reshape(3,-1)
        if nn_type=='FCN':
            # the dim of data: (channel x height x width)
            datapoints.append(torch.tensor(datapoint, dtype=float).view(-1).to(device))
        elif nn_type=='CNN':
            # the dim of data: channel x height x width
            # TODO: not to hard code the dimensnion for CNN
            datapoints.append(torch.tensor(datapoint.T, dtype=float).view(1, -1, 3).to(device))
    
    return datapoints

def get_datapoints_pro(aug_data, ref, window_size, ds, device, nn_type):
    '''
    to extract datapoints of each time step from the padded reference trajectory with mimic online updates

    Args:
        aug_data: a list of padded reference trajectories in the order of online updating, with the last element as the final valid reference trajectory
        ref: a list of time steps indicating when online updates happened
        window_size: the whole length of the window
        ds: the stride for down sampling inside the window
        device: where to send datapoints for training later
        nn_type: the type of the neural network we used for training, which determines the dimension of datapoints
    Returns:
        datapoints: extracted datapoints
    '''
    datapoints = []

    l = aug_data[-1].shape[1] - window_size + 1
    for k in range(l):
        choice = 0
        for ele in ref:
            if k>ele:
                choice += 1
            else:
                break
        datapoint = aug_data[choice][0:3, k:k+window_size:ds].reshape(3,-1)         
        if nn_type=='FCN':
            # the dim of data: (channel x height x width)
            datapoints.append(torch.tensor(datapoint, dtype=float).view(-1).to(device))
        elif nn_type=='CNN':
            # the dim of data: channel x height x width
            # TODO: not to hard code the dimensnion for CNN
            datapoints.append(torch.tensor(datapoint.T, dtype=float).view(1, -1, 3).to(device))
    
    return datapoints

def get_dataset(datapoints, batch_size):
    '''
    to construct a dataset from datapoints

    Args:
        datapoints: individual datapoints
        batch_size: how many datapoints to batch together
    Returns:
        dataset: the constructed dataset
    '''
    dataset = []
    
    l = len(datapoints)
    idx = 0
    while idx + batch_size - 1 < l:
        data = datapoints[idx:idx+batch_size]
        # the dim of the element in dataset: batchsize x (channel x height x width) for FCN and batchsize x channel x height x width for CNN
        batch = torch.stack(data)
        dataset.append(batch)
        idx += batch_size

    return dataset

def get_prediction(datapoint, block_list, y):
    '''
    to get the trainable contorl inputs
    '''
    dataset = get_dataset(datapoint=datapoint, batch_size=y.shape[1])
    u = np.zeros(y.shape)
    for dof,block in enumerate(block_list):
        if block is None:
            continue
        else:
            block.eval()
            try:
                u[dof, :] = block(dataset[0]).cpu().detach().numpy().flatten()
            except:
                u[dof, :] = block(dataset[0].float()).cpu().detach().numpy().flatten()
    return u

def wandb_plot(i_iter, frequency, t_stamp, ff, fb, y, theta_, t_stamp_list, theta_list, T_go_list, p_int_record, obs_ago, obs_ant, des_ago, des_ant):
    '''
    to plot the ff input, the fb input and the tracking performance in joint space
    '''
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

        ### added pressure fig
        fig2 = plt.figure(figsize=(18, 18))

        p_position0 = fig2.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressures of Dof_0')
        line = []
        line_temp, = p_position0.plot(t_stamp, obs_ago[0,:], linewidth=2, label=r'obs_ago')
        line.append( line_temp )
        line_temp, = p_position0.plot(t_stamp, obs_ant[0,:], linewidth=2, label=r'obs_ant')
        line.append( line_temp )
        line_temp, = p_position0.plot(t_stamp, des_ago[0,:], linewidth=2, label=r'des_ago')
        line.append( line_temp )
        line_temp, = p_position0.plot(t_stamp, des_ant[0,:], linewidth=2, label=r'des_ant')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
            
        p_position1 = fig2.add_subplot(312)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressures of Dof_1')
        line = []
        line_temp, = p_position1.plot(t_stamp, obs_ago[1,:], linewidth=2, label=r'obs_ago')
        line.append( line_temp )
        line_temp, = p_position1.plot(t_stamp, obs_ant[1,:], linewidth=2, label=r'obs_ant')
        line.append( line_temp )
        line_temp, = p_position1.plot(t_stamp, des_ago[1,:], linewidth=2, label=r'des_ago')
        line.append( line_temp )
        line_temp, = p_position1.plot(t_stamp, des_ant[1,:], linewidth=2, label=r'des_ant')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
        
        p_position2 = fig2.add_subplot(313)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressures of Dof_2')
        line = []
        line_temp, = p_position2.plot(t_stamp, obs_ago[2,:], linewidth=2, label=r'obs_ago')
        line.append( line_temp )
        line_temp, = p_position2.plot(t_stamp, obs_ant[2,:], linewidth=2, label=r'obs_ant')
        line.append( line_temp )
        line_temp, = p_position2.plot(t_stamp, des_ago[2,:], linewidth=2, label=r'des_ago')
        line.append( line_temp )
        line_temp, = p_position2.plot(t_stamp, des_ant[2,:], linewidth=2, label=r'des_ant')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Pressures Monitoring'+' Iter '+str(i_iter+1))
        plots.append(wandb.Image(plt, caption="matplotlib image"))                
        
        wandb.log({'related plots': plots})
        plt.close()

def get_grads_list(dataset, block_list, additional=False):
    '''
    Nt x nff
    '''
    X_list = []
    Y_list = []

    for idx,block in enumerate(block_list):
        if block is None:
            X_list.append(None)
            Y_list.append(None)
            continue

        block.train()
        for param in block.parameters():
            if param.grad is None:
                break
            param.grad.zero_()

        flag = True
        if len(dataset)==1:
            chosen_dataset = dataset[0]
        else:
            chosen_dataset = dataset[idx]

        for data in chosen_dataset:
            grad = []

            if additional:
                data.requires_grad = True
            
            try:
                block(data.float()).mean().backward()
            except:
                block(data).mean().backward()                       
            for param in block.parameters():
                grad.append(torch.clone(param.grad.cpu().view(-1)))
                param.grad.zero_()
            
            if additional:
                gradsY = torch.clone(data.grad.cpu().view(-1))
                data.grad.zero_()

                gradsY_ = np.copy(gradsY.reshape(1, -1)) if flag else np.concatenate((gradsY_, gradsY.reshape(1, -1)), axis=0)

            grads = torch.cat(grad)            
            grads_ = np.copy(grads.reshape(1, -1)) if flag else np.concatenate((grads_, grads.reshape(1, -1)), axis=0)
            flag = False if flag else False

        X_list.append(grads_)
        if additional:
            Y_list.append(gradsY_)

    if additional:
        return X_list, Y_list
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

if __name__=='__main__':
    print('to be done')