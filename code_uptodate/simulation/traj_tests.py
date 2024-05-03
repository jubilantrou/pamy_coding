# %% import libraries
import PAMY_CONFIG
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import random
from get_handle import get_handle
import o80_pam
import torch
from Trainable_blocks import *
import torch.nn as nn
import time
from RealRobotGeometry import RobotGeometry
import time
import torch.nn as nn
import o80
import wandb

# %% set parameters (TODO: consider to organize as an independent file later)
obj             = 'sim'                 # training for the simulator or for the real robot
coupling        = 'yes'                 # if to use the references of all degrees of freedom as the input for each CNN, in consideration of the coupling
nr_channel      = 1                     # 1 channel for p, which we consider for now, while 3 channels for p, v and a, regarding of the input for CNN
h               = 10                    # the extension time point length in both directions
nr_iteration    = 600                   # training iterations
width           = 3                     # the width of the input for CNN, indicating using the references of all 3 degrees of freedom
ds              = 1                     # the stride when construct the input
height          = int((2*h)/ds)+1       # the height of the input for CNN
filter_size     = 7                     # the kernel size for height dimension in CNN
learning_rate   = [0.01, 0.1, 0.01]       # learning rates
seed = 5431
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %% initialize the gpu and the robot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
print('device: {}'.format(device))

if obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
else:
    raise ValueError('The variable obj needs to be assigned either as sim or as real!')

Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
RG = RobotGeometry()

# %% create functions
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_random():
    theta = np.zeros(3)
    theta[0] = random.choice([random.randrange(-700, -250)/10, random.randrange(250, 700)/10])
    theta[1] = random.randrange(150, 750)/10
    theta[2] = random.randrange(150, 750)/10
    t        = random.randrange(90, 100)/100
    theta    = theta * math.pi/180
    return (t, theta)


def get_compensated_data(data=None):
    I_left = np.tile(data[:, 0].reshape(-1, 1), (1, h))
    I_right = np.tile(data[:, -1].reshape(-1, 1), (1, h))
    y_ = np.hstack((I_left, data, I_right))
    return y_    

def get_dataset(y, batch_size=1):
    l = y.shape[1]
    y_ = get_compensated_data(y)
    data = []
    dataset = []

    for k in range(l):
        y_temp = np.concatenate((y_[0, k:k+2*h+1:ds].reshape(-1,1), y_[1, k:k+2*h+1:ds].reshape(-1,1), y_[2, k:k+2*h+1:ds].reshape(-1,1)), axis=1)
        # data: channel x height x width
        data.append(torch.tensor(y_temp, dtype=float).view(nr_channel, height, width).to(device))

    idx = 0
    while idx + batch_size - 1 < l:
        data_ = data[idx:idx+batch_size]
        batch = torch.stack(data_)
        # elements in dataset: batchsize x channel x height x width
        dataset.append(batch)
        idx += batch_size

    return dataset

def get_grads_list(dataset, cnn_list):
    X_list = []

    for cnn in cnn_list:
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

def get_step_size(nr, dof, step_size_version='constant'):
    factor = [0.1, 0.1, 0.1]
    constant_list = np.copy(learning_rate)
    if step_size_version == 'constant':
        step_size = constant_list[dof]
    elif step_size_version == 'sqrt':
        step_size = factor[dof]/(2+np.sqrt(nr))
    return step_size 

def set_parameters(W_list, cnn_list, idx_list, shape_list):
    for dof in range(3):
        W = W_list[dof]
        cnn = cnn_list[dof]
        i = 0
        for param in cnn.parameters():
            idx_1 = idx_list[i]
            idx_2 = idx_list[i+1]
            W_ = torch.tensor(W[idx_1:idx_2]).view(shape_list[i])
            param.data = W_.to(device)
            i += 1
    return cnn_list

def get_prediction(cnn_list, y, denorm):
    dataset = get_dataset(y, batch_size=y.shape[1])
    u = np.zeros(y.shape)
    for dof in range(3):
        cnn = cnn_list[dof]
        cnn.eval()
        try:
            u[dof, :] = cnn(dataset[0]).cpu().detach().numpy().flatten()
        except:
            u[dof, :] = cnn(dataset[0].float()).cpu().detach().numpy().flatten()
    return u

# %% define the cnn
cnn_list   = []
name_list  = [] # 2*layers
shape_list = [] # 2*layers
idx_list   = [] # 2*layers+1
idx = 0
idx_list.append(idx)

def weight_init(l):
    if isinstance(l,nn.Conv2d) or isinstance(l,nn.Linear):
        nn.init.xavier_normal_(l.weight,gain=0.1)
        nn.init.normal_(l.bias)

for dof in range(3):
    cnn = CNN(channel_in=nr_channel, filter_size=filter_size, height=height, width=width)
    # temp = torch.load('/home/mtian/Desktop/MPI-intern/training_log_temp/180/' + str(dof))
    # cnn.load_state_dict(temp)
    # cnn.apply(weight_init)
    cnn.to(device)
    cnn_list.append(cnn)

for name, param in cnn.named_parameters():
    name_list.append(name)
    shape_list.append(param.shape)
    d_idx = len(param.data.view(-1))
    idx += d_idx
    idx_list.append(idx)

print('the number of trainable parameters: {}'.format(idx_list[-1]))
root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_1/linear_model' + '/1'
mkdir(root_model_epoch) 
for dof in range(3):
    cnn = cnn_list[dof]
    root_file = root_model_epoch + '/' + str(dof)
    print('paras of {}: {}'.format(dof, cnn.state_dict()))
    torch.save(cnn.state_dict(), root_file)

# %% do the online learning
seed = 3154
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

### for penalty parameters testing
rec = 0

fig1 = plt.figure(figsize=(18, 18))
ax1_position0 = fig1.add_subplot(111)
plt.xlabel(r'Time $t$ in s')
plt.ylabel(r'Dof_0')
line = []

for i in range(200):
    (t, angle) = get_random()
    (p, v, a, j, theta, t_stamp, _) = RG.PathPlanning(
        time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, part=0)
    
    line_temp, = ax1_position0.plot(t_stamp, theta[0, :]/math.pi*180, linewidth=1)
    line.append( line_temp )

line_temp, = ax1_position0.plot(t_stamp, [120]*len(t_stamp), linewidth=1)
line.append( line_temp )
line_temp, = ax1_position0.plot(t_stamp, [-120]*len(t_stamp), linewidth=1)
line.append( line_temp )

plt.legend(handles=line, shadow=True)
plt.show()
###

# i_iter = 0
# while 1:
#     print('------------')
#     print('iter {}'.format(i_iter))

#     # (t, angle) = get_random()
#     # print(t)
#     # print(angle)
#     t = 0.97
#     angle = [0.581, 0.794, 1.037]
#     (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method=2)
#     theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
#     theta_ = np.copy(theta)
#     theta = theta - theta[:, 0].reshape(-1, 1)
#     Pamy.ImportTrajectory(theta, t_stamp)

#     for idx,ele in enumerate(p_int_record):
#         print('target_{}: {}'.format(idx+1,RG.AngleToEnd(ele, frame='Cartesian')))

#     '''
#     Below codes are used to plot the information about the reference trajectory and the real trajectory.
#     '''
#     # TODO: not exactly the same starting point
#     # theta_ starts with pos_init while y starts without
#     # have added, but still to check the starting point choice and the pressure given style
#     plots = []
#     if_plot = 1
#     if_both = 0
#     if_joint = 0
#     if_cylinder = 0

#     if if_plot:
#         legend_position = 'best'
#         fig = plt.figure(figsize=(18, 18))

#         ax_position0 = fig.add_subplot(311)
#         plt.xlabel(r'Time $t$ in s')
#         plt.ylabel(r'Position of Dof_0 in degree')
#         line = []
#         line_temp, = ax_position0.plot(t_stamp, theta_[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof0_des')
#         line.append( line_temp )
#         for j in range(len(theta_list)):
#             line_temp, = ax_position0.plot(t_stamp_list[j], theta_list[j][0, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof0_traj_candidate_'+str(j+1))
#             line.append( line_temp )
#             line_temp, = ax_position0.plot(T_go_list[j], p_int_record[j][0] * 180 / math.pi, 'o', label='target_'+str(j+1))
#             line.append( line_temp )
#         plt.legend(handles=line, loc=legend_position, shadow=True)
            
#         ax_position1 = fig.add_subplot(312)
#         plt.xlabel(r'Time $t$ in s')
#         plt.ylabel(r'Position of Dof_1 in degree')
#         line = []
#         line_temp, = ax_position1.plot(t_stamp, theta_[1, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof1_des')
#         line.append( line_temp )
#         for j in range(len(theta_list)):
#             line_temp, = ax_position1.plot(t_stamp_list[j], theta_list[j][1, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof1_traj_candidate_'+str(j+1))
#             line.append( line_temp )
#             line_temp, = ax_position1.plot(T_go_list[j], p_int_record[j][1] * 180 / math.pi, 'o', label='target_'+str(j+1))
#             line.append( line_temp )
#         plt.legend(handles=line, loc=legend_position, shadow=True)
        
#         ax_position2 = fig.add_subplot(313)
#         plt.xlabel(r'Time $t$ in s')
#         plt.ylabel(r'Position of Dof_2 in degree')
#         line = []
#         line_temp, = ax_position2.plot(t_stamp, theta_[2, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof2_des')
#         line.append( line_temp )
#         for j in range(len(theta_list)):
#             line_temp, = ax_position2.plot(t_stamp_list[j], theta_list[j][2, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof2_traj_candidate_'+str(j+1))
#             line.append( line_temp )
#             line_temp, = ax_position2.plot(T_go_list[j], p_int_record[j][2] * 180 / math.pi, 'o', label='target_'+str(j+1))
#             line.append( line_temp )
#         plt.legend(handles=line, loc=legend_position, shadow=True)

#         plt.suptitle('Joint Space Trajectory Tracking Performance'+' Iter '+str(i_iter))
#         # plots.append(wandb.Image(plt, caption="matplotlib image"))                
#         plt.show()
