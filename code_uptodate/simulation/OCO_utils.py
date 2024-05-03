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
import pathlib
from typing import Tuple, Any
from nptyping import NDArray, Shape

def mkdir(path: pathlib.Path) -> None:
    '''
    to check if a given file exists, and to create it if not 

    Args:
        path: the path of the file to be checked
    '''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path) 

def fix_seed(seed: int) -> None:
    '''
    to ensure the reproducibility

    Args:
        seed: the chosen seed to be fixed
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_random() -> Tuple[float, NDArray[Shape['3, ...'], np.float64]]:
    '''
    to randomly generate the mimic interception time point and the mimic interception position

    Returns:
        t: the mimic interception time point, in seconds
        theta: the mimic interception position, in rads and in the joint space
    '''
    t = random.randrange(90, 100)/100
    theta = np.zeros(3)
    # theta[0] = random.choice([random.randrange(-900, -450)/10, random.randrange(450, 900)/10])
    # theta[1] = random.randrange(450, 800)/10
    # theta[2] = random.randrange(150, 750)/10
    theta[0] = random.randrange(-900, -450)/10
    theta[1] = random.randrange(450, 800)/10
    theta[2] = random.randrange(200, 750)/10
    theta = theta * math.pi/180
    return (t, theta)

def get_compensated_data(data: NDArray[Any, np.float64], h_l: int, h_r: int) -> NDArray[Any, np.float64]:
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
        # TODO: try to normalize the input data, and to sample not uniformly
        datapoint = aug_data[choice][0:3, k:k+window_size:ds].reshape(3,-1)
        # datapoint = ((np.hstack((aug_data[choice][0:3, k:k+45:9], aug_data[choice][0:3, k+45:k+56:1], aug_data[choice][0:3, k+56:k+141:6])))*np.array([[1/np.pi],[2/np.pi],[2/np.pi]])).reshape(3,-1)    
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

def get_prediction(datapoints, block_list):
    '''
    Returns:
        u: the trainble feedforward control inputs
    '''
    Nt = len(datapoints)
    dataset = get_dataset(datapoints=datapoints, batch_size=Nt)
    u = np.zeros((3, Nt))

    for dof, block in enumerate(block_list):
        if block is None:
            continue
        else:
            block.eval()
            try:
                u_temp = block(dataset[0]).cpu().detach().numpy()
            except:
                u_temp = block(dataset[0].float()).cpu().detach().numpy()
            # SISO: there are seperate networks for different DoFs
            if u_temp.shape[1] == 1:
                u[dof, :] = u_temp.flatten()
            # MIMO: there is only one network for all DoFs
            elif u_temp.shape[1] == 3:
                u = u_temp.T
    
    return u

def get_par_paiff_par_wff(datapoints, block_list, num_paras):
    # TODO: need to change the inputs if using different network structures for different DoF later
    '''
    to get the gradient information about par_paiff_par_wff from the trainable feedforward control policies

    Args:
        datapoints: extracted datapoints from the reference trajectory
        block_list: a list of networks for the trainable feedforward control policies
        num_paras: the number of trainable parameters for doing dimension checking
    Returns:
        pay_paiff_par_wff_list: a list of gradient information about par_paiff_par_wff, corresponding to the block_list
    '''
    dataset = get_dataset(datapoints=datapoints, batch_size=1)
    par_paiff_par_wff_list = []

    for idx, block in enumerate(block_list):
        if block is None:
            par_paiff_par_wff_list.append(None)
            continue

        block.train()
        for param in block.parameters():
            if param.grad is None:
                break
            param.grad.zero_()

        flag = True
        for data in dataset:
            grad = []
         
            try:
                pai_temp = block(data.float())
            except:
                pai_temp = block(data)
            out_dim = pai_temp.shape[1]
            
            for idx_out in range(out_dim):
                # SISO: there are seperate blocks and each of them outputs 1-dim control input
                if out_dim == 1:
                    pai_temp.mean().backward()
                # MIMO: there is only one block and it outputs 3-dim control inputs
                elif out_dim == 3:
                    pai_temp[0, idx_out].backward()
                else:
                    raise ValueError("Something is wrong with the output dimension of the block.")
                
                for param in block.parameters():
                    grad.append(torch.clone(param.grad.cpu().view(-1)))
                    param.grad.zero_()

            grad_ele = torch.cat(grad)
            
            grads = np.copy(grad_ele.reshape(out_dim, -1)) if flag else np.concatenate((grads, grad_ele.reshape(out_dim, -1)), axis=0)
            flag = False

        assert grads.shape[0] == (out_dim*len(datapoints)), 'Something is wrong with the dimension of par_paiff_par_wff.'
        assert grads.shape[1] == num_paras, 'Something is wrong with the dimension of par_paiff_par_wff.'
        par_paiff_par_wff_list.append(grads)

    return par_paiff_par_wff_list

def get_par_paifb_par_wfb_and_ydiff(block_list, fb_datapoints_list, num_paras):
    # TODO: need to change the inputs if using different network structures for different DoF later
    '''
    to get the gradient information about par_paifb_par_wfb and par_paifb_par_ydiff from the trainable feedback control policies

    Args:
        block_list: a list of networks for the trainable feedback control policies
        fb_datapoints_list: a list of datapoints given to the trainable feedback control policies during trajectory tracking, corresponding to the block_list
        num_paras: the number of trainable parameters for doing dimension checking
    Returns:
        par_paifb_par_wfb_list: a list of gradient information about par_paifb_par_wfb, corresponding to the block_list
        par_paifb_par_ydiff_list: a list of gradient information about par_paifb_par_ydiff, corresponding to the block_list
    '''
    par_paifb_par_wfb_list = []
    par_paifb_par_ydiff_list = []

    for idx, block in enumerate(block_list):
        if block is None:
            par_paifb_par_wfb_list.append(None)
            par_paifb_par_ydiff_list.append(None)
            continue

        block.train()
        for param in block.parameters():
            if param.grad is None:
                break
            param.grad.zero_()

        flag = True
        for data in fb_datapoints_list[idx]:
            grad_wfb = []
            data.requires_grad = True
            if data.grad is not None:
                data.grad.zeros_()
            
            try:
                block(data.float()).mean().backward()
            except:
                block(data).mean().backward()

            for param in block.parameters():
                grad_wfb.append(torch.clone(param.grad.cpu().view(-1)))
                param.grad.zero_()
            grad_ele_wfb = torch.cat(grad_wfb)           
            grads_wfb = np.copy(grad_ele_wfb.reshape(1, -1)) if flag else np.concatenate((grads_wfb, grad_ele_wfb.reshape(1, -1)), axis=0)

            grad_ydiff = torch.clone(data.grad.cpu().view(-1))
            data.grad.zero_()
            grads_ydiff = np.copy(grad_ydiff.reshape(1, -1)) if flag else np.concatenate((grads_ydiff, grad_ydiff.reshape(1, -1)), axis=0)

            flag = False

        assert grads_wfb.shape[0] == len(fb_datapoints_list[idx]), 'Something is wrong with the dimension of par_paifb_par_wfb.'
        assert grads_wfb.shape[1] == num_paras, 'Something is wrong with the dimension of par_paifb_par_wfb.'
        par_paifb_par_wfb_list.append(grads_wfb)

        assert grads_ydiff.shape[0] == len(fb_datapoints_list[idx]), 'Something is wrong with the dimension of par_paifb_par_ydiff.'
        assert grads_ydiff.shape[1] == 10, 'Something is wrong with the dimension of par_paifb_par_ydiff.'
        par_paifb_par_ydiff_list.append(grads_ydiff)

    return par_paifb_par_wfb_list, par_paifb_par_ydiff_list

def set_parameters(block_list, W_list, idx_list, shape_list, device):
    # TODO: need to change the inputs if using different network structures for different DoF later
    '''
    to update the trainable parameters for the trainable control policies

    Args:
        block_list: a list of networks for the trainable control policies
        W_list: a list of newly acquired parameter values to be updated, corresponding to the block_list
        idx_list: a list recording the accumulated number of named parameters for the block
        shape_list: a list recording the shape of each named parameter for the block
        device: where are the networks sent to work
    Returns:
        block_list: the list of networks for the updated trainable control policies
    '''
    for idx, block in enumerate(block_list):
        if block is None:
            continue

        W = W_list[idx]
        i = 0
        for param in block.parameters():
            idx_1 = idx_list[i]
            idx_2 = idx_list[i+1]
            W_ele = torch.tensor(W[idx_1:idx_2]).view(shape_list[i])
            param.data = W_ele.to(device)
            i += 1
    return block_list

def get_decreasing_step_size(iter, lr_list):
    '''
    to get the decreasing step size for the current iteration using the square root

    Args:
        iter: the current iteration
        lr_list: a list of the initial constant step sizes
    Returns:
        step_size_list: the list of the decreased step sizes
    '''
    step_size_list = []
    for lr in lr_list:
        step_size_list.append(lr/(1+np.sqrt(iter)))
    return step_size_list
