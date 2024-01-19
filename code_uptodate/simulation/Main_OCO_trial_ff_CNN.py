'''
This script is used to train the robot with online convex optimization.
control diagram description: only feedforward
ff control policy description: CNN + online gradient descent
training data description: multiple trajectories with updates in h/f and the same initial state
'''
# %% import
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
from CNN import CNN
import torch.nn as nn
import time
from RealRobotGeometry import RobotGeometry
import time
import torch.nn as nn
import o80
import wandb
# %% parameter configuration file (can be organized as an independent file later)
obj = 'sim'
# parameters about the simulator setting in get_handle.py
# %% initialize the gpu and the robot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
else:
    raise ValueError('The variable obj needs to be assigned either as sim or as real!')

Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
RG = RobotGeometry()
# sigma = np.zeros(4)
# for dof in range(4):
#     sigma[dof] = np.max([PAMY_CONFIG.pressure_ago_ub[dof], np.abs(PAMY_CONFIG.pressure_ago_lb[dof])])
# max_val = PAMY_CONFIG.pressure_ago_ub / sigma 
# min_val = PAMY_CONFIG.pressure_ago_lb / sigma 
# %% constant
# alpha_list        = [1e-5, 1e-5, 1e-5]
# epsilon_list      = [1e-17, 1e-17, 1e-17]
# learning_mode     = 'b'
# mode_name         = 'ff'
# step_size_version = 'constant'
coupling          = 'yes'
nr_epoch          = 1
nr_channel        = 1
h                 = 50
nr_iteration      = 5
# version           = 'random'
# train_index       = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
# test_index        = [30, 39] #, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
# root              = '/home/hao/Desktop/MPI/Pamy_simulation'
# folder_name       = version + '_' + 'ep' + '_' + str(nr_epoch) + '_' + 'h' + '_' + str(h) + '_' + 'st' + '_' + step_size_version + '_' + str(len(train_index))
# root_data         = root + '/' + 'data' + '/' + 'oco_multi' + '/' + 'cnn_fc' + '/'  + folder_name
# root_learning     = root_data + '/' + 'learning'
# root_verify       = root_data + '/' + 'verify'
# root_model        = root_data + '/' + 'model'

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

# mkdir(root_data)
# mkdir(root_learning)
# mkdir(root_verify)
# mkdir(root_model)

# gamma       = 0.3
width      = 3
ds          = 1
height       = int((2*h)/ds)+1
filter_size = 11
# learning_rate = [1e-4, 1e-5, 3e-5]
# weight_decay = 0.00
# %% functions
def get_random():
    theta = np.zeros(3)
    theta[0] = random.choice([random.randrange(-750, -250)/10, random.randrange(250, 750)/10])
    theta[1] = random.choice([random.randrange(100, 350)/10, random.randrange(550, 800)/10])
    theta[2] = random.choice([random.randrange(100, 350)/10, random.randrange(550, 800)/10])
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
        y_temp = np.concatenate((y_[0, k:k+2*h+1:ds], y_[1, k:k+2*h+1:ds], y_[2, k:k+2*h+1:ds]))
        '''
        batchsize x channel x height x width
        '''
        data.append(torch.tensor(y_temp, dtype=float).view(nr_channel, height, width).to(device))

    idx = 0
    while idx + batch_size - 1 < l:
        data_ = data[idx:idx+batch_size]
        batch = torch.stack(data_)
        dataset.append(batch)
        idx += batch_size
    return dataset

def get_grads_list(dataset, cnn_list):
    X_list = []
    dof = 0
    for cnn in cnn_list:
        for name, param in cnn.named_parameters():
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
            
            
            for name, param in cnn.named_parameters():
                grad.append(torch.clone(param.grad.cpu().view(-1)))
                param.grad.zero_()
            
            # [grad.append(param.grad.view(-1)) for name, param in cnn.named_parameters()]
            grads = torch.cat(grad)
            grads_ = np.copy(grads.reshape(1, -1)) if flag else np.concatenate((grads_, grads.reshape(1, -1)), axis=0)
            flag = False if flag else False
        X_list.append(grads_)
        dof += 1
    return X_list

def get_step_size(nr, dof, step_size_version='constant'):
    factor = [1.0, 1.0, 1.0]
    # constant_list = [0.2, 0.2, 0.5]
    constant_list = [3e-1, 3e-1, 3e-1]
    if step_size_version == 'constant':
        step_size = constant_list[dof]
    elif step_size_version == 'sqrt':
        step_size = factor[dof]/(2+np.sqrt(nr))
        # if dof == 1:
        #     step_size = 2.0
    return step_size 

# def verify(Pamys, index_list, path, name='train', b_list=None):
#     root_verify = path + '/' + name
#     mkdir(root_verify)
#     root_ff = root_verify + '/' + 'ff'
#     mkdir(root_ff)
#     root_fb = root_verify + '/' + 'fb'
#     mkdir(root_fb)

#     for i in range(len(Pamys)):
#         Pamy    = Pamys[i]
#         get_verify(Pamy, b_list, root_ff, index=index_list[i], mode_name='ff')
#         get_verify(Pamy, b_list, root_fb, index=index_list[i], mode_name='ff+fb')

def set_parameters(W_list, cnn_list, idx_list, shape_list):
    for dof in range(3):
        W = W_list[dof]
        cnn = cnn_list[dof]
        i = 0
        for name, param in cnn.named_parameters():
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
            u[dof, :] = cnn(dataset[0]).cpu().detach().numpy().flatten() * denorm[dof]
        except:
            u[dof, :] = cnn(dataset[0].float()).cpu().detach().numpy().flatten() * denorm[dof]
    return u
# %% initilization
print('init begins')
Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
# Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
obs_pressure = np.array(frontend.latest().get_observed_pressures())
pressure_ago = obs_pressure[:, 0]
pressure_ant = obs_pressure[:, 1]
print(pressure_ago)
print(pressure_ant)
print('init done')
# %% define the cnn
cnn_list   = []
name_list  = []
shape_list = []
idx_list   = []
idx = 0
idx_list.append(idx)

def weight_init(l):
    if isinstance(l,nn.Conv2d) or isinstance(l,nn.Linear):
        nn.init.uniform_(l.weight,a=-0.01,b=0.01)
        nn.init.constant_(l.bias,0.01)

for dof in range(3):
    cnn = CNN(filter_size=filter_size, width=width, height=height, channel_in=nr_channel)
    cnn.apply(weight_init)
    cnn.to(device)
    cnn_list.append(cnn)

for name, param in cnn.named_parameters():  # models are the same for all dofs
    name_list.append(name)
    shape_list.append(param.shape)
    d_idx = len(param.data.view(-1))
    idx += d_idx
    idx_list.append(idx)
# print('para name')
# print(name_list)
# print('para shape')
# print(shape_list)
print('idx list')
print(idx_list)
# %% Learning
# traj_house = []
# t_house = []
# angle_house = []
# for num in range(nr_iteration):
#     (t, angle) = get_random()
#     t_house.append(t)
#     angle_house.append(angle)
#     (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)
#     (_, p_int) = RG.AngleToEnd(angle=angle,frame='Cartesian')
#     traj_ele = {}
#     traj_ele['t'] = t
#     traj_ele['angle'] = angle
#     traj_ele['p'] = p
#     traj_ele['v'] = v
#     traj_ele['a'] = a
#     traj_ele['j'] = j
#     traj_ele['theta'] = theta
#     traj_ele['t_stamp'] = t_stamp
#     traj_ele['p_int'] = p_int
#     traj_house.append(traj_ele)
# np.save('/home/mtian/training_log_temp/t_house.npy',t_house)
# np.save('/home/mtian/training_log_temp/angle_house.npy',angle_house)

# (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)
# theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
# (_, p_int) = RG.AngleToEnd(angle=angle,frame='Cartesian')
# print('p_int')
# print(p_int.reshape(-1).tolist())
# hit_point = handle.frontends["hit_point"]
# hit_point.add_command(p_int.reshape(-1).tolist(),(0,0,0),o80.Duration_us.seconds(1),o80.Mode.QUEUE)
# hit_point.pulse_and_wait()
# for j in range(len(t_stamp)):
#     joints = theta[:,j].reshape(-1).tolist()
#     frontend.add_command(joints,(0,0,0,0),o80.Duration_us.milliseconds(100),o80.Mode.QUEUE)
#     frontend.pulse_and_wait()

# wandb.init(
#     entity='jubilantrou',
#     project='pamy_oco_trial'
# )

for i_epoch in range(nr_epoch):
    # Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
    (t, angle) = get_random()
    (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, time_update_record) = RG.updatedPathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)
    # (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)

    print('targets:')
    for ele in p_int_record:
        print(ele/math.pi*180)
    print('time interval:')
    for ele in t_stamp_list:
        print(ele[0], ele[-1])
    print('update moments:')
    for ele in time_update_record:
        print(ele)

    # t = traj_house[i_it]['t']
    # angle = traj_house[i_it]['angle']
    # p = traj_house[i_it]['p']
    # v = traj_house[i_it]['v']
    # a = traj_house[i_it]['a']
    # j = traj_house[i_it]['j']
    # theta = traj_house[i_it]['theta']
    # t_stamp = traj_house[i_it]['t_stamp']
    # p_int = traj_house[i_it]['p_int']

    # print('target:')
    # print(angle/math.pi*180)
    # (_, p_int) = RG.AngleToEnd(angle=angle,frame='Cartesian')
    # hit_point = handle.frontends["hit_point"]
    # hit_point.add_command(p_int.reshape(-1).tolist(),(0,0,0),o80.Duration_us.seconds(1),o80.Mode.QUEUE)
    # hit_point.pulse_and_wait()

    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = theta
    # print('part traj overview')
    # print(np.min(theta,axis=1))
    # print(np.max(theta,axis=1))
    theta = theta - theta[:, 0].reshape(-1, 1)
    Pamy.ImportTrajectory(theta, t_stamp)  #  import the desired trajectories and the time stamp
    Pamy.GetOptimizer_convex(angle_initial_read, nr_channel=nr_channel, coupling=coupling)
    
    # index = get_index(train_index, index_used)
    # index_used.append(index)

    # Pamy    = Pamy_train[train_index.index(index)]  # find the position of index in train_index
    # t_stamp = Pamy.t_stamp

    u = get_prediction(cnn_list, Pamy.y_desired, PAMY_CONFIG.pressure_limit)

    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(u, coupling=coupling, learning_mode='u')
    def get_difference(x):
        y = np.zeros(x.shape)
        for i in range(1, y.shape[1]):
            y[:, i] = (x[:, i]-x[:, i-1])/(t_stamp[i]-t_stamp[i-1])
        return y
    v_y = get_difference(y)
    a_y = get_difference(v_y)
    y_cylinder = np.zeros((3,y.shape[1]))
    for i in range(y.shape[1]):
        (_,end) = RG.AngleToEnd(y[:3,i], frame='Cylinder')
        y_cylinder[:,i] = end
    # y = np.hstack([angle_initial_read.reshape(4,1), y[:,:-1]])
    # print('observer for pressure here!')
    # print(obs_ago[0,::10])
    # print(obs_ant[0,::10])
    y_out = y - y[:, 0].reshape(-1, 1)

    # plot the reference trajectory and the real trajectory

    # TODO: not exactly the same starting point
    # theta_ starts with pos_init while y starts without
    # have added, but still to check the starting point choice and the pressure given style

    if_plot = 1
    if_both = 0
    if_joint = 0
    if_cylinder = 0
    plots = []

    if if_plot:
        legend_position = 'best'
        fig = plt.figure(figsize=(18, 18))

        ax_position0 = fig.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position of Dof_0 in degree')
        line = []
        line_temp, = ax_position0.plot(t_stamp, y[0, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,10)), label=r'Pos_Dof0_out')
        line.append( line_temp )
        line_temp, = ax_position0.plot(t_stamp, theta_[0, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,10)), label=r'Pos_Dof0_des')
        line.append( line_temp )
        for j in range(len(theta_list)):
            line_temp, = ax_position0.plot(t_stamp_list[j], theta_list[j][0, :] * 180 / math.pi, linewidth=2, label='Dof0_traj_candidate_'+str(j+1))
            line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
            
        ax_position1 = fig.add_subplot(312)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position of Dof_1 in degree')
        line = []
        line_temp, = ax_position1.plot(t_stamp, y[1, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,10)), label=r'Pos_Dof1_out')
        line.append( line_temp )
        line_temp, = ax_position1.plot(t_stamp, theta_[1, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,10)), label=r'Pos_Dof1_des')
        line.append( line_temp )
        for j in range(len(theta_list)):
            line_temp, = ax_position1.plot(t_stamp_list[j], theta_list[j][1, :] * 180 / math.pi, linewidth=2, label='Dof1_traj_candidate_'+str(j+1))
            line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
        
        ax_position2 = fig.add_subplot(313)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position of Dof_2 in degree')
        line = []
        line_temp, = ax_position2.plot(t_stamp, y[2, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,10)), label=r'Pos_Dof2_out')
        line.append( line_temp )
        line_temp, = ax_position2.plot(t_stamp, theta_[2, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,10)), label=r'Pos_Dof2_des')
        line.append( line_temp )
        for j in range(len(theta_list)):
            line_temp, = ax_position2.plot(t_stamp_list[j], theta_list[j][2, :] * 180 / math.pi, linewidth=2, label='Dof2_traj_candidate_'+str(j+1))
            line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Joint Space Trajectory Tracking Performance'+' Iter ')
        # plots.append(wandb.Image(plt, caption="matplotlib image"))                
        plt.show()

        if if_both:
            fig = plt.figure(figsize=(18, 18))

            ax_position00 = fig.add_subplot(311)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Position of angle $\theta$ in degree')
            line = []
            line_temp, = ax_position00.plot(t_stamp, y_cylinder[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_$\theta$_out')
            line.append( line_temp )
            line_temp, = ax_position00.plot(t_stamp, p[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_$\theta$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)
                
            ax_position11 = fig.add_subplot(312)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Position of radius $\eta$ in m')
            line = []
            line_temp, = ax_position11.plot(t_stamp, y_cylinder[1, :], linewidth=2, label=r'Pos_$\eta$_out')
            line.append( line_temp )
            line_temp, = ax_position11.plot(t_stamp, p[1, :], linewidth=2, label=r'Pos_$\eta$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)
            
            ax_position22 = fig.add_subplot(313)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Position of height $\xi$ in degree')
            line = []
            line_temp, = ax_position22.plot(t_stamp, y_cylinder[2, :], linewidth=2, label=r'Pos_$\xi$_out')
            line.append( line_temp )
            line_temp, = ax_position22.plot(t_stamp, p[2, :], linewidth=2, label=r'Pos_$\xi$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            plt.suptitle('Cylinder Space Trajectory Tracking Performance'+' Iter '+str(i_it))
            plots.append(wandb.Image(plt, caption="matplotlib image"))                
            # plt.show()

        if if_joint:
            fig = plt.figure(figsize=(18, 18))

            ax_velocity0 = fig.add_subplot(121)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Vel in rad/s')
            line = []
            line_temp, = ax_velocity0.plot(t_stamp[1:], v_y[0, 1:], linewidth=2, label=r'Vel_Dof0_out')
            line.append( line_temp )
            line_temp, = ax_velocity0.plot(t_stamp[1:], v_y[1, 1:], linewidth=2, label=r'Vel_Dof1_out')
            line.append( line_temp )
            line_temp, = ax_velocity0.plot(t_stamp[1:], v_y[2, 1:], linewidth=2, label=r'Vel_Dof2_out')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            ax_acceleration0 = fig.add_subplot(122)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Acc in rad/s^2')
            line = []
            line_temp, = ax_acceleration0.plot(t_stamp[1:], a_y[0, 1:], linewidth=2, label=r'Acc_Dof0_out')
            line.append( line_temp )
            line_temp, = ax_acceleration0.plot(t_stamp[1:], a_y[1, 1:], linewidth=2, label=r'Acc_Dof1_out')
            line.append( line_temp )
            line_temp, = ax_acceleration0.plot(t_stamp[1:], a_y[2, 1:], linewidth=2, label=r'Acc_Dof2_out')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            plt.suptitle('Joint Space Real Data'+' Iter '+str(i_it))                
            plots.append(wandb.Image(plt, caption="matplotlib image"))
            # plt.show()

        if if_cylinder:
            fig = plt.figure(figsize=(18, 18))

            ax_velocity1 = fig.add_subplot(231)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Vel in rad/s')
            line = []
            line_temp, = ax_velocity1.plot(t_stamp, v[0, :], linewidth=2, label=r'Vel_$\theta$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            ax_acceleration1 = fig.add_subplot(232)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Acc in rad/s^2')
            line = []
            line_temp, = ax_acceleration1.plot(t_stamp, a[0, :], linewidth=2, label=r'Acc_$\theta$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            ax_jerk1 = fig.add_subplot(233)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Jerk in rad/s^3')
            line = []
            line_temp, = ax_jerk1.plot(t_stamp, j[0, :], linewidth=2, label=r'Jerk_$\theta$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            ax_velocity2 = fig.add_subplot(234)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Vel in m/s')
            line = []
            line_temp, = ax_velocity2.plot(t_stamp, v[1, :], linewidth=2, label=r'Vel_$\eta$_des')
            line.append( line_temp )
            line_temp, = ax_velocity2.plot(t_stamp, v[2, :], linewidth=2, label=r'Vel_$\xi$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            ax_acceleration2 = fig.add_subplot(235)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Acc in m/s^2')
            line = []
            line_temp, = ax_acceleration2.plot(t_stamp, a[1, :], linewidth=2, label=r'Acc_$\eta$_des')
            line.append( line_temp )
            line_temp, = ax_acceleration2.plot(t_stamp, a[2, :], linewidth=2, label=r'Acc_$\xi$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            ax_jerk2 = fig.add_subplot(236)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Jerk in m/s^3')
            line = []
            line_temp, = ax_jerk2.plot(t_stamp, j[1, :], linewidth=2, label=r'Jerk_$\eta$_des')
            line.append( line_temp )
            line_temp, = ax_jerk2.plot(t_stamp, j[2, :], linewidth=2, label=r'Jerk_$\xi$_des')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            plt.suptitle('Cylinder Space Desired Data'+' Iter '+str(i_it))
            plots.append(wandb.Image(plt, caption="matplotlib image"))                
            # plt.show()
        
        wandb.log({'traj related plots': plots})

    # root_file = root_epoch + '/' + str(i_it) 
    # file = open(root_file, 'wb')
    # pickle.dump(t_stamp, file, -1)
    # pickle.dump(t_stamp, file, -1)
    # pickle.dump(angle_initial_read, file, -1)
    # pickle.dump(y, file, -1)
    # pickle.dump(Pamy.y_desired, file, -1)
    # pickle.dump(ff, file, -1)
    # pickle.dump(fb, file, -1)
    # pickle.dump(obs_ago, file, -1)
    # pickle.dump(obs_ant, file, -1)
    # pickle.dump(W_list, file, -1)
    # pickle.dump(index, file, -1)
    # file.close()

    print('begin {},{}. optimization'.format(i_epoch, i_it))
    # nr_round = i_epoch * len(train_index) + i_it + 1 + 5
    W_list = [None] * 3
    for i in range(3):    
        W = []
        [W.append(param.data.view(-1)) for param in cnn_list[i].parameters()]
        W = torch.cat(W)
        W_list[i] = W.cpu().numpy().reshape(-1, 1)
    part3 = get_grads_list(get_dataset(Pamy.y_desired), cnn_list)
    part2 = [Pamy.O_list[i].Bu for i in PAMY_CONFIG.dof_list]
    part1_temp = y-theta_
    part1 = [part1_temp[i].reshape(1,-1) for i in range(len(part1_temp))]
    loss = [np.linalg.norm(part1_temp[i].reshape(1,-1)) for i in range(len(part1_temp))]
    step_list = [5e-3, 5e-3, 5e-3]
    for dof in range(3):  # update the linear model b
        W_list[dof] = W_list[dof] - step_list[dof]*PAMY_CONFIG.pressure_limit[dof]*(part1[dof]@part2[dof]@part3[dof]).reshape(-1, 1)
    #     '''
    #     b = b - s_k * pinv(1/t*sum(L.T * L)+alpha/t*sum(X.T * X)+epsilon*I) * L.T * (y_out - y_des)
    #     '''

    #     [hessian, gradient, sum_1_list[dof], sum_2_list[dof], X_list] = get_newton_method(nr_round, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list, cnn_list)
    #     sk                                                            = get_step_size(nr_round, dof, step_size_version=step_size_version)
    #     W_list[dof]                                                   = get_update(W_list[dof], sk, hessian, gradient)
    #     # W_list[dof]                                                   = get_projection(dof, W_list[dof], X_list[dof])
    cnn_list = set_parameters(W_list, cnn_list, idx_list, shape_list)
    print('end {},{}. optimization'.format(i_epoch, i_it))
    print('loss:')
    print(loss)
    # wandb.log({'loss_0': loss[0], 'loss_1': loss[1], 'loss_2': loss[2]}, i_it+1)

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    # Pamy.PressureInitialization()
    angle_initial_read =np.array(frontend.latest().get_positions())
    print('prepared for next training')
    print('____________')
    print('____________')

# index_used = [] 
# root_model_epoch = root_model + '/' + str(i_epoch)  # save the model at each epoch
# mkdir(root_model_epoch)
# wandb.finish() 
# for dof in range(3):
#     cnn = cnn_list[dof]
#     root_file = '/home/mtian/training_log_temp/model' + str(i_epoch) + str(dof)
#     mkdir(root_file)
#     torch.save(cnn.state_dict(), root_file)