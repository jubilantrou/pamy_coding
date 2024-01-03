#!/usr/bin/env python3
'''
This script is used to plot figures for online convex optimization
'''
import numpy as np
import pickle5 as pickle
import math
import matplotlib.pyplot as plt
import os
import PAMY_CONFIG
# %% 
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
# %% constants
train_index       = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
test_index        = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]

'''
trajectories with similar shapes
'''
# train_index = [11, 15, 17, 23] #, 32, 33, 37, 41, 47, 56, 58, 62, 66]
# test_index  = [7, 13, 30, 45]

legend_position   = 'upper right'
dof_list          = [0,1,2]
saving_mode       = 'on'
projection        = 'osqp'
learning_mode     = 'b'
mode_name         = 'ff'
coupling          = 'yes'
step_size_version = 'sqrt'
nr_channel        = 1
# %% folders
learning_mode     = 'b'
mode_name         = 'ff'
step_size_version = 'constant'
coupling          = 'yes'
nr_epoch          = 50
nr_channel        = 3
h                 = 100
version           = 'bank'
nr_train          = 1

root            = '/home/hao/Desktop/Learning'
folder_name     = version + '_' + 'ep' + '_' + str(nr_epoch) + '_' + 'h' + '_' + str(h) + '_' + 'st' + '_' + step_size_version + '_' + str(nr_train)
# folder_name     = learning_mode + '_' + mode_name  + '_' + step_size_version + '_' + projection + '_' + coupling + '_' + str(nr_channel) + 'channel'
root_data       = root + '/' + 'data' + '/' + 'oco_multi' + '/' + 'cnn_fc' + '/' + folder_name
data_verify     = root_data + '/' + 'verify'
data_model      = root_data + '/' + 'model'
data_learning   = root_data + '/' + 'learning'
root_figure     = root + '/'  + 'figure' + '/' + 'oco_multi' + '/' + folder_name
figure_learning = root_figure + '/' + 'learning'
figure_verify   = root_figure + '/' + 'verify'
figure_error    = root_figure + '/' + 'error'
mkdir(figure_learning)
mkdir(figure_verify)
mkdir(figure_error)
# %% functions
def get_data(path=None):
    # open file and read the data
    f = open(path, 'rb')
    t_stamp       = pickle.load(f)
    t             = pickle.load(f)
    angle_initial = pickle.load(f)
    y             = pickle.load(f)
    y_des         = pickle.load(f)
    ff            = pickle.load(f)
    fb            = pickle.load(f)
    obs_ago       = pickle.load(f)
    obs_ant       = pickle.load(f)
    b_list        = pickle.load(f)
    f.close()

    y_des = y_des + angle_initial.reshape(-1, 1)
    # for i in range(ff.shape[0]):
    #     ff[i, ff[i, :]>limit_max[i]] = limit_max[i]
    #     ff[i, ff[i, :]<limit_min[i]] = limit_min[i]
    return (t_stamp, t, y, y_des, ff, fb, obs_ago, obs_ant)

def AngleToEnd(angle):
    '''angle is absolute'''
    l_1 = 0.38
    l_2 = 0.40
    theta0 = angle[0]
    theta1 = angle[1]
    theta2 = angle[2]

    x = math.cos(theta0) * math.sin(theta1) * l_1
    y = math.sin(theta0) * math.sin(theta1) * l_1
    z = math.cos(theta1) * l_1
    position_A = np.array([x, y, z])
    x = math.cos(theta0) * math.sin(theta1) * l_1 + math.cos(theta0) * math.sin(theta1 + theta2) * l_2
    y = math.sin(theta0) * math.sin(theta1) * l_1 + math.sin(theta0) * math.sin(theta1 + theta2) * l_2
    z = math.cos(theta1) * l_1 + math.cos(theta1 + theta2) * l_2
    position_B = np.array([x, y, z])

    return (position_A, position_B)

def get_traj(y):
    l = y.shape[1]
    traj = np.zeros((3, l))
    for i in range(l):
        (_, traj[:, i]) = AngleToEnd(y[:, i])
    return traj

def get_error(y, y_des):
    l          = y.shape[1]
    p          = get_traj(y)
    p_des      = get_traj(y_des)
    error_mean = np.zeros(4)
    error_max  = np.zeros(4)

    for i in range(3):
        error_mean[i] = np.linalg.norm(y[i, :]-y_des[i, :], ord=1)/l
        error_max[i]  = np.max(np.abs(y[i, :]-y_des[i, :]))
    
    error_     = 0
    for i in range(l):
        error_ = np.linalg.norm(p[:, i]-p_des[:, i], ord=2)
        error_mean[3] += error_
        if error_ > error_max[3]:
            error_max[3] = error_

    error_mean[3] = error_mean[3]/l
    return (error_mean, error_max)

def plot_iteration(data_path, figure_path):
    (t_stamp, t, y, y_des, ff, fb, obs_ago, obs_ant) = get_data(data_path)
    l1 = len(t_stamp)
    l2 = y_des.shape[1]
    if l1 > l2:
        l = l2
    else:
        l = l1

    t_stamp = t_stamp[0:l]
    y = y[:, 0:l]
    y_des = y_des[:, 0:l]
    ff = ff[:, 0:l]
    obs_ago = obs_ago[:, 0:l]
    obs_ant = obs_ant[:, 0:l]

    if saving_mode == 'on':
        fig, axs = plt.subplots(3, 3, figsize=(40, 20))
        for dof in range(3):
            line = []
            ax = axs[0, dof]
            ax.set_xlabel(r'Time $t$ in s')
            ax.set_ylabel(r'Error $\Delta \theta$ in degree')
            line_temp, = ax.plot(t_stamp, y_des[dof, :]*180/math.pi, linewidth=1.5, linestyle='--', label=r'des')
            line.append(line_temp)
            line_temp, = ax.plot(t_stamp, y[dof, :]*180/math.pi, label=r'output')
            line.append(line_temp)
            ax.grid()
            ax.legend(handles=line, loc=legend_position)

            line = []
            ax = axs[1, dof]
            ax.set_xlabel(r'Time $t$ in s')
            ax.set_ylabel(r'Normalized pressure')
            line_temp, = ax.plot(t_stamp, ff[dof, :], linewidth=2.0, linestyle='-', label=r'feedforward')
            line.append(line_temp)
            ax.axhline(PAMY_CONFIG.pressure_max[dof], color='b')
            ax.axhline(PAMY_CONFIG.pressure_min[dof], color='b') 
            ax.grid()
            ax.legend(handles=line, loc=legend_position)

            line = []
            ax = axs[2, dof]
            ax.set_xlabel(r'Time $t$ in s')
            ax.set_ylabel(r'Normalized pressure')
            line_temp, = ax.plot(t_stamp, obs_ago[dof, :], label=r'ago')
            line.append(line_temp)
            ax.axhline(PAMY_CONFIG.ago_max_list[dof], color='b')
            ax.axhline(PAMY_CONFIG.ago_min_list[dof], color='b') 
            ax.grid()
            ax.legend(handles=line, loc=legend_position)
        plt.savefig(figure_path + '.pdf')
        plt.close()

    [error_mean, error_max] = get_error(y, y_des)
    return (error_mean, error_max)

def plot_epoch(data_path, figure_path):
    files        = os.listdir(data_path)
    nr_iteration = len(files)
    error_mean   = np.zeros((nr_iteration, 4))
    error_max    = np.zeros((nr_iteration, 4))

    for i in range(nr_iteration):
        root_data   = data_path + '/' + str(i)
        root_figure = figure_path + '/' + str(i)
        [error_mean[i, :], error_max[i, :]] = plot_iteration(data_path=root_data, figure_path=root_figure)

    return (error_mean, error_max, nr_iteration)
    
def plot_learning(data_path, figure_path):
    files            = os.listdir(data_path)
    figure_learning  = figure_path[0]
    figure_error     = figure_path[1]
    nr_epoch         = len(files)
    it_list          = []
    total_it         = 0
    it_list.append(total_it)
    error_epoch_mean =  np.zeros((nr_epoch, 4))
    error_epoch_max  =  np.zeros((nr_epoch, 4))

    for i in range(nr_epoch):
        root_data   = data_path + '/' + str(i)
        root_figure = figure_learning + '/' + str(i)
        mkdir(root_figure)
        [error_mean, error_max, nr_iteration] = plot_epoch(data_path=root_data, figure_path=root_figure)
        if i == 0:
            error_mean_list = np.copy(error_mean)
            error_max_list  = np.copy(error_max)
        else:
            error_mean_list = np.vstack((error_mean_list, error_mean))
            error_max_list  = np.vstack((error_max_list, error_max))
        total_it += nr_iteration
        it_list.append(total_it-1)
        error_epoch_mean[i, :] = np.mean(error_mean, axis=0)
        error_epoch_max[i, :]  = np.mean(error_max, axis=0)
    
    fig, axs = plt.subplots(4, 1, figsize=(40, 20))
    for dof in range(3):
        line = []
        ax = axs[dof]
        ax.set_xlabel(r'Iterations')
        ax.set_ylabel(r'Error $\delta \theta$ in degree')
        line_temp, = ax.plot(range(total_it), error_mean_list[:, dof]*180/math.pi, label=r'dof-{}, mean'.format(dof))
        line.append(line_temp)
        line_temp, = ax.plot(range(total_it), error_max_list[:, dof]*180/math.pi, label=r'dof-{}, max'.format(dof))
        line.append(line_temp)
        for i in range(len(it_list)):
            ax.axvline(x=it_list[i], color = 'b')
        
        ax.grid()
        ax.legend(handles=line, loc=legend_position)
    
    line = []
    ax = axs[3]
    ax.set_xlabel(r'Iterations')
    ax.set_ylabel(r'Error $\Delta s$ in m')
    line_temp, = ax.plot(range(total_it), error_mean_list[:, 3], label=r'mean')
    line.append(line_temp)
    line_temp, = ax.plot(range(total_it), error_max_list[:, 3], label=r'max')
    line.append(line_temp)
    for i in range(len(it_list)):
        ax.axvline(x=it_list[i], color = 'b')
    ax.grid()
    ax.legend(handles=line, loc=legend_position)
    # plt.show()
    plt.savefig(figure_error + '/' + 'learning' + '.pdf')
    plt.close()

    fig, axs = plt.subplots(4, 1, figsize=(40, 20))
    for dof in range(3):
        line = []
        ax = axs[dof]
        ax.set_xlabel(r'Epoches')
        ax.set_ylabel(r'Error $\delta \theta$ in degree')
        line_temp, = ax.plot(range(nr_epoch), error_epoch_mean[:, dof]*180/math.pi, '--', marker='x', markersize=20, label=r'dof-{}, mean'.format(dof))
        line.append(line_temp)
        line_temp, = ax.plot(range(nr_epoch), error_epoch_max[:, dof]*180/math.pi, '--', marker='x', markersize=20, label=r'dof-{}, max'.format(dof))
        line.append(line_temp)        
        ax.grid()
        ax.legend(handles=line, loc=legend_position)
    
    line = []
    ax = axs[3]
    ax.set_xlabel(r'Epoches')
    ax.set_ylabel(r'Error $\Delta s$ in m')
    line_temp, = ax.plot(range(nr_epoch), error_epoch_mean[:, 3], '--', marker='x', markersize=20, label=r'mean')
    line.append(line_temp)
    line_temp, = ax.plot(range(nr_epoch), error_epoch_max[:, 3], '--', marker='x', markersize=20, label=r'max')
    line.append(line_temp) 
    ax.grid()
    ax.legend(handles=line, loc=legend_position)
    # plt.show()
    plt.savefig(figure_error + '/' + 'epoch_error' + '.pdf')
    plt.close()

def plot_ff(data_path, figure_path, name):
    files      = os.listdir(data_path)
    nr_file    = len(files)
    error_mean = np.zeros((nr_file, 4))
    error_max  = np.zeros((nr_file, 4))
    for i in range(nr_file):
        if name == 'train':
            index = train_index[i]
        else:
            index = test_index[i]
        root_data   = data_path + '/' + str(index)
        root_figure = figure_path + '/' + str(index)
        [error_mean[i, :], error_max[i, :]] = plot_iteration(data_path=root_data, figure_path=root_figure)
    return (error_mean, error_max, nr_file)

def plot_tt(data_path, figure_path, name):
    root_data   = data_path + '/' + 'ff'
    root_figure = figure_path + '/' + 'ff'
    mkdir(root_figure)
    [error_ff_mean, error_ff_max, nr] = plot_ff(root_data, root_figure, name)

    root_data = data_path + '/' + 'fb'
    root_figure = figure_path + '/' + 'fb'
    mkdir(root_figure)
    [error_fb_mean, error_fb_max, nr] = plot_ff(root_data, root_figure, name)
    
    return (error_ff_mean, error_ff_max, error_fb_mean, error_fb_max, nr)

def plot_verify(data_path, figure_path):
    figure_verify = figure_path[0]
    figure_error  = figure_path[1]
    root_data = data_path + '/' + 'train'
    root_figure = figure_verify + '/' + 'train'
    mkdir(root_figure)
    [error_ff_train_mean, error_ff_train_max, error_fb_train_mean, error_fb_train_max, nr_train] = plot_tt(root_data, root_figure, name='train')

    # root_data = data_path + '/' + 'test'
    # root_figure = figure_verify + '/' + 'test'
    # mkdir(root_figure)
    # [error_ff_test_mean, error_ff_test_max, error_fb_test_mean, error_fb_test_max, nr_test] = plot_tt(root_data, root_figure, name='test')

    # error_ff_mean = np.concatenate((error_ff_train_mean, error_ff_test_mean), axis=0)
    # error_ff_max  = np.concatenate((error_ff_train_max, error_ff_test_max), axis=0)
    # error_fb_mean = np.concatenate((error_fb_train_mean, error_fb_test_mean), axis=0)
    # error_fb_max  = np.concatenate((error_fb_train_max, error_fb_test_max), axis=0)
    # nr            = nr_train + nr_test

    # fig, axs = plt.subplots(4, 2, figsize=(40, 20))
    # for dof in range(3):
    #     line = []
    #     ax = axs[dof, 0]
    #     ax.set_xlabel(r'Iterations')
    #     ax.set_ylabel(r'Error $\delta \theta$ in degree')
    #     line_temp, = ax.plot(range(nr), error_ff_mean[:, dof]*180/math.pi, label=r'dof-{}, mean ff'.format(dof))
    #     line.append(line_temp)
    #     line_temp, = ax.plot(range(nr), error_ff_max[:, dof]*180/math.pi, label=r'dof-{}, max ff'.format(dof))
    #     line.append(line_temp)
    #     ax.axvline(x=nr_train, color = 'b')
    #     ax.grid()
    #     ax.legend(handles=line, loc=legend_position)

    #     line = []
    #     ax = axs[dof, 1]
    #     ax.set_xlabel(r'Iterations')
    #     ax.set_ylabel(r'Error $\delta \theta$ in degree')
    #     line_temp, = ax.plot(range(nr), error_fb_mean[:, dof]*180/math.pi, label=r'dof-{}, mean fb'.format(dof))
    #     line.append(line_temp)
    #     line_temp, = ax.plot(range(nr), error_fb_max[:, dof]*180/math.pi, label=r'dof-{}, max fb'.format(dof))
    #     line.append(line_temp)
    #     ax.axvline(x=nr_train, color = 'b')
    #     ax.grid()
    #     ax.legend(handles=line, loc=legend_position)
    
    # line = []
    # ax = axs[3, 0]
    # ax.set_xlabel(r'Iterations')
    # ax.set_ylabel(r'Error $\Delta s$ in m')
    # line_temp, = ax.plot(range(nr), error_ff_mean[:, 3], label=r'mean ff')
    # line.append(line_temp)
    # line_temp, = ax.plot(range(nr), error_ff_max[:, 3], label=r'max ff')
    # line.append(line_temp)
    # ax.axvline(x=nr_train, color = 'b')
    # ax.grid()
    # ax.legend(handles=line, loc=legend_position)

    # line = []
    # ax = axs[3, 1]
    # ax.set_xlabel(r'Iterations')
    # ax.set_ylabel(r'Error $\Delta s$ in m')
    # line_temp, = ax.plot(range(nr), error_fb_mean[:, 3], label=r'mean fb')
    # line.append(line_temp)
    # line_temp, = ax.plot(range(nr), error_fb_max[:, 3], label=r'max fb')
    # line.append(line_temp)
    # ax.axvline(x=nr_train, color = 'b')
    # ax.grid()
    # ax.legend(handles=line, loc=legend_position)

    # plt.savefig(figure_error + '/' + 'verify' + '.pdf')
    # plt.close()

# %% main
plot_learning(data_path=data_learning, figure_path=(figure_learning, figure_error))
# plot_verify(data_path=data_verify, figure_path=(figure_verify, figure_error))