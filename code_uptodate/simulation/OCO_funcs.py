'''
This script is used to define the related functions 
for the OCO training procedure.
'''
import math
import os
import numpy as np
import pickle5 as pickle
import random
import torch

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
            # the dim of each datapoint: (channel x height x width)
            datapoints.append(torch.tensor(datapoint, dtype=float).view(-1).to(device))
        elif nn_type=='CNN':
            # the dim of each datapoint: channel x height x width
            # TODO: do not hard code the dimensnion for CNN
            # TODO: check the way to combine the references from 3 DoFs
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
            # the dim of each datapoint: (channel x height x width)
            datapoints.append(torch.tensor(datapoint, dtype=float).view(-1).to(device))
        elif nn_type=='CNN':
            # the dim of each datapoint: channel x height x width
            # TODO: do not hard code the dimensnion for CNN
            # TODO: check the way to combine the references from 3 DoFs
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
        # the dim of each element in dataset: batchsize x (channel x height x width) for FCN and batchsize x channel x height x width for CNN
        batch = torch.stack(data)
        dataset.append(batch)
        idx += batch_size

    return dataset

def get_prediction(datapoints, block_list, Nt):
    '''
    to get the trainable feedforward contorl inputs by feeding datapoints from the reference trajectory into the trainable feedforward contorl policy

    Args:
        datapoints: extracted datapoints from the reference trajectory
        block_list: a list of networks for the trainable control policies
        Nt: the number of time steps
    Returns:
        u: the trainble feedforward control inputs
    '''
    dataset = get_dataset(datapoints=datapoints, batch_size=Nt)
    u = np.zeros((3, Nt))
    for dof,block in enumerate(block_list):
        if block is None:
            continue
        else:
            block.eval()
            try:
                u_temp = block(dataset[0]).cpu().detach().numpy()
            except:
                u_temp = block(dataset[0].float()).cpu().detach().numpy()
            # SISO: seperate networks for different DoF
            if u_temp.shape[1] == 1:
                u[dof, :] = u_temp.flatten()
            # MIMO: one network for all DoFs
            elif u_temp.shape[1] == 3:
                u = u_temp.T
    return u

# TODO: mark to continue reformatting
def get_grads_list(dataset, block_list, additional=False):
    '''
    3*Nt x nff
    to get needed gradient information related to trainable control policies, par_paiff_par_wff for feedforward case
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
            
            for ele in range(3):
                try:
                    # block(data.float()).mean().backward()
                    block(data.float())[0,ele].backward()
                except:
                    # block(data).mean().backward()
                    block(data)[0,ele].backward()                       
                for param in block.parameters():
                    grad.append(torch.clone(param.grad.cpu().view(-1)))
                    param.grad.zero_()

            grads = torch.cat(grad)            
            grads_ = np.copy(grads.reshape(3, -1)) if flag else np.concatenate((grads_, grads.reshape(3, -1)), axis=0)
            flag = False if flag else False

        X_list.append(grads_)

    return X_list

def get_grads_list_fb(dataset, block_list, additional=False):
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