'''
This script is used to compare the performance of ILC and OCO (u and b).
'''
import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import os
import math
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
# %% constant
train_index     = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
test_index      = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
anchor_ago_list = np.array([17500, 20700, 16000, 15000])
anchor_ant_list = np.array([17500, 16300, 16000, 15000])
ago_max_list    = np.array([22000, 25000, 22000, 22000])
ant_max_list    = np.array([22000, 23000, 22000, 22000])
ago_min_list    = [13000, 13500, 10000, 8000]
ant_min_list    = [13000, 14500, 10000, 8000]
limit_max       = ago_max_list - anchor_ago_list
limit_min       = ago_min_list - anchor_ago_list
root            = '/home/hao/Desktop/Learning'
root_data       = root + '/' + 'data' + '/' + 'comparison'
root_figure     = root + '/' + 'figure' + '/' + 'comparison'
mkdir(root_figure)
index           = 17
# %% functions
def get_data_ILC(index, data_path):
    root_file = data_path + '/' + str(index) + '_ILC'
    f            = open(root_file, 'rb')
    t_stamp      = pickle.load(f)
    t_list       = pickle.load(f)
    angle_initial = pickle.load(f)
    y_history    = pickle.load(f)
    repeated     = pickle.load(f)
    y_pid        = pickle.load(f)
    ff_history   = pickle.load(f)
    fb_history   = pickle.load(f)
    ago_history  = pickle.load(f)
    ant_history  = pickle.load(f)
    d_history    = pickle.load(f)
    P_history    = pickle.load(f)
    d_lifted_history = pickle.load(f)
    P_lifted_history = pickle.load(f)
    f.close()
    y_des = y_history[0] + angle_initial.reshape(-1, 1)
    y_list = y_history[1:]
    return (t_stamp, y_des, y_list, ff_history, ago_history, ant_history)

def get_data_OCO(index, data_path)
# %% read data
[t_stamp, y_des_ILC, y_list_ILC, ff_ILC, ago_ILC, ant_ILC] = get_data_ILC(index, root_data)
[t_stamp, y_des_OCO_b, y_list_OCO_b, ff_OCO_b, ago_OCO_b, ant_OCO_b] = get_data_OCO(index, root_data, mode='b')
[t_stamp, y_des_OCO_u, y_list_OCO_u, ff_OCO_u, ago_OCO_u, ant_OCO_u] = get_data_OCO(index, root_data, mode='u')
# %% main