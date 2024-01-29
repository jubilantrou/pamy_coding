'''
This script is used to train the robot for better tracking performance with online convex optimization.
* [control diagram description] trainable feedforward block and fixed feedback block
* [ff control policy description] (CNN +) FCN + online gradient descent
* [fb control policy description] fixed PD controller (PID controller for doing AngleInitialization)
* [training data description] multiple reference trajectories at the same initial state, with updates 
w/o (w/) a delay of h/f to mimic possible online replanning
'''
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
from CNN import CNN
import torch.nn as nn
import time
from RealRobotGeometry import RobotGeometry
import torch.nn as nn
import o80
import wandb
from pyinstrument import Profiler

# %% set parameters (TODO: to organize the procedure as an independent file later)
obj                   = 'sim'                                   # training for the simulator or for the real robot
coupling              = 'yes'                                   # if to use the references of all degrees of freedom as the input for each CNN, in consideration of the coupling
nr_channel            = 1                                       # 1 channel for p, which we consider for now, while 3 channels for p, v and a, regarding of the input for CNN
h                     = 25                                      # the extension time point length in both directions
nr_iteration          = 1000                                    # training iterations
width                 = 3                                       # the width of the input for CNN, indicating using the references of all 3 degrees of freedom
ds                    = 1                                       # the stride when construct the input
height                = int((2*h)/ds)+1                         # the height of the input for CNN
filter_size           = 7                                       # the kernel size for height dimension in CNN
learning_rate         = np.array([1.0e-2, 1.0e-1, 1.0e-2])      # learning rates
seed                  = 5431                                    # chosen seed for the reproducibility
flag_wandb            = True                                    # if to enable wandb for recording the training process
flag_time_analysis    = False                                   # if to use pyinstrument to analyse the time consumption of different parts
flag_time_record      = False                                   # if to show the used time of each iteration
save_path_num         = 3                                       # the number at the end of the path where we store results
flag_0_dof            = False                                   # if to include Dof0 in the training procedure, and we set it to False for the simulator as there is a dead zone for Dof0
method_updating_traj  = 2                                       # the method on how to update the trajectory, 1: update w/ a delay of h=10; 2: update w/o the delay

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

# %% create functions (TODO: to organize often used ones as an independent script)
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_random():
    theta = np.zeros(3)
    theta[0] = random.choice([random.randrange(-700, -250)/10, random.randrange(250, 700)/10])
    theta[1] = random.randrange(150, 750)/10
    theta[2] = random.randrange(150, 750)/10
    t        = random.randrange(90, 100)/100
    theta    = theta * math.pi/180
    return (t, theta)


def get_compensated_data(data=None, option=None):
    I_left = np.tile(data[:, 0].reshape(-1, 1), (1, h))
    I_right = np.tile(data[:, -1].reshape(-1, 1), (1, h))
    if option=='only_left':
        y_ = np.hstack((I_left, data))
    else:
        y_ = np.hstack((I_left, data, I_right))
    return y_    

def get_dataset(y, batch_size=1, option=None, ref=None):
    l = y.shape[1]
    y_ = get_compensated_data(y)
    data = []
    dataset = []

    for k in range(l):
        if option is None:
            y_temp = np.concatenate((y_[0, k:k+2*h+1:ds].reshape(-1,1), y_[1, k:k+2*h+1:ds].reshape(-1,1), y_[2, k:k+2*h+1:ds].reshape(-1,1)), axis=1)
        else:
            choice = 0
            for ele in ref:
                if k>ele:
                    choice += 1
                else:
                    break
            if choice<len(ref):
                y_temp = np.concatenate((option[choice][0, k:k+2*h+1:ds].reshape(-1,1), option[choice][1, k:k+2*h+1:ds].reshape(-1,1), option[choice][2, k:k+2*h+1:ds].reshape(-1,1)), axis=1)
            else:
                y_temp = np.concatenate((y_[0, k:k+2*h+1:ds].reshape(-1,1), y_[1, k:k+2*h+1:ds].reshape(-1,1), y_[2, k:k+2*h+1:ds].reshape(-1,1)), axis=1)            
        # data: (channel x height x width)
        data.append(torch.tensor(y_temp, dtype=float).view(-1).to(device))

    idx = 0
    while idx + batch_size - 1 < l:
        data_ = data[idx:idx+batch_size]
        batch = torch.stack(data_)
        # elements in dataset: batchsize x (channel x height x width)
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
    for dof in range(len(cnn_list)):
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

def get_prediction(cnn_list, y, option=None, ref=None):
    t_begin = time.time()
    dataset = get_dataset(y, batch_size=y.shape[1], option=option, ref=ref)
    u = np.zeros(y.shape)
    num = len(cnn_list)
    if num==2:
        for dof in range(num):
            cnn = cnn_list[dof]
            cnn.eval()
            try:
                u[dof+1, :] = cnn(dataset[0]).cpu().detach().numpy().flatten()
            except:
                u[dof+1, :] = cnn(dataset[0].float()).cpu().detach().numpy().flatten()
    elif num==3:
        for dof in range(num):
            cnn = cnn_list[dof]
            cnn.eval()
            try:
                u[dof, :] = cnn(dataset[0]).cpu().detach().numpy().flatten()
            except:
                u[dof, :] = cnn(dataset[0].float()).cpu().detach().numpy().flatten()
    t_end = time.time()
    t_used = t_end - t_begin
    return u, t_used

# %% define the cnn
cnn_list   = []
name_list  = []
shape_list = []
idx_list   = []
idx = 0
idx_list.append(idx)
if flag_0_dof:
    num_nn = 3
else:
    num_nn = 2

def weight_init(l):
    if isinstance(l,nn.Conv2d) or isinstance(l,nn.Linear):
        nn.init.xavier_normal_(l.weight,gain=0.1)
        nn.init.normal_(l.bias)

for dof in range(num_nn):
    cnn = CNN(channel_in=nr_channel, filter_size=filter_size, height=height, width=width)
    ###
    # for potential pre-trained weights loading
    ###
    # temp = torch.load('/home/mtian/Desktop/MPI-intern/training_log_temp/180/' + str(dof))
    # cnn.load_state_dict(temp)

    ###
    # for self-defined method of weights initialization
    ###
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

root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_' + str(save_path_num) + '/linear_model' + '/1'
mkdir(root_model_epoch) 
for dof in range(num_nn):
    cnn = cnn_list[dof]
    root_file = root_model_epoch + '/' + str(dof)
    # print('paras of {}: {}'.format(dof, cnn.state_dict()))
    torch.save(cnn.state_dict(), root_file)

# %% do the online learning
###
# to visualize the target point 'p_int' and the planned trajectory 'theta' in the simulator
###
# hit_point = handle.frontends["hit_point"]
# hit_point.add_command(p_int.reshape(-1).tolist(),(0,0,0),o80.Duration_us.seconds(1),o80.Mode.QUEUE)
# hit_point.pulse_and_wait()
# for j in range(len(t_stamp)):
#     joints = theta[:,j].reshape(-1).tolist()
#     frontend.add_command(joints,(0,0,0,0),o80.Duration_us.milliseconds(100),o80.Mode.QUEUE)
#     frontend.pulse_and_wait()

if flag_wandb:
    wandb.init(
        entity='jubilantrou',
        project='pamy_oco_trial'
    )

fix_seed(seed)
i_iter = 0
t_inf = 0
while 1:
    if flag_time_analysis:
        profiler = Profiler()
        profiler.start()

    if flag_time_record:
        t_begin = time.time()

    print('------------')
    print('iter {}'.format(i_iter+1))

    (t, angle) = get_random()
    (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
        time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method=method_updating_traj)
    
    aug_ref_traj = []
    for j in range(len(update_point_index_list)):
        comp = get_compensated_data(np.hstack((theta[:,:update_point_index_list[j]], theta_list[j][:,time_update_record[j]:time_update_record[j]+h+1])), option='only_left')
        aug_ref_traj.append(comp)

    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = np.copy(theta)
    theta = theta - theta[:, 0].reshape(-1, 1)
    Pamy.ImportTrajectory(theta, t_stamp)

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    angle_initial_read = np.array(frontend.latest().get_positions())
    Pamy.GetOptimizer_convex(angle_initial=angle_initial_read, h=h, nr_channel=nr_channel, coupling=coupling)
    if method_updating_traj==1:
        u, t_used = get_prediction(cnn_list, Pamy.y_desired)
    elif method_updating_traj==2:
        u, t_used = get_prediction(cnn_list, Pamy.y_desired, aug_ref_traj, update_point_index_list)
    t_inf += t_used

    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list=u, mode_name='ff+fb', coupling=coupling, learning_mode='u')
    y_out = y - y[:, 0].reshape(-1, 1)

    if (i_iter+1)%20==0:
        root_file = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(save_path_num) +'/linear_model/log_data/' + str(i_iter+1)
        mkdir(root_file)
        file = open(root_file + '/log.txt', 'wb')
        pickle.dump(t_stamp, file, -1)
        pickle.dump(angle_initial_read, file, -1)
        pickle.dump(y, file, -1)
        pickle.dump(Pamy.y_desired, file, -1)
        pickle.dump(ff, file, -1)
        pickle.dump(fb, file, -1)
        pickle.dump(obs_ago, file, -1)
        pickle.dump(obs_ant, file, -1)
        file.close()

    print('pressure check:')
    print('dof 0: {} ~ {}'.format(min(ff[0,:]),max(ff[0,:])))
    print('dof 1: {} ~ {}'.format(min(ff[1,:]),max(ff[1,:])))
    print('dof 2: {} ~ {}'.format(min(ff[2,:]),max(ff[2,:])))
    
    if_plot_pressure = 1
    plots = []
    if if_plot_pressure and (i_iter+1)%50==0:
        legend_position = 'best'
        fig1 = plt.figure(figsize=(18, 18))

        ax1_position0 = fig1.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_0')
        line = []
        line_temp, = ax1_position0.plot(t_stamp, u[0, :], linewidth=2, label=r'uff_Dof0')
        line.append( line_temp )
        line_temp, = ax1_position0.plot(t_stamp, fb[0, :], linewidth=2, label=r'ufb_Dof0')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
            
        ax1_position1 = fig1.add_subplot(312)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_1')
        line = []
        line_temp, = ax1_position1.plot(t_stamp, u[1, :], linewidth=2, label=r'uff_Dof1')
        line.append( line_temp )
        line_temp, = ax1_position1.plot(t_stamp, fb[1, :], linewidth=2, label=r'ufb_Dof1')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
        
        ax1_position2 = fig1.add_subplot(313)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_2')
        line = []
        line_temp, = ax1_position2.plot(t_stamp, u[2, :], linewidth=2, label=r'uff_Dof2')
        line.append( line_temp )
        line_temp, = ax1_position2.plot(t_stamp, fb[2, :], linewidth=2, label=r'ufb_Dof2')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Pressure Input'+' Iter '+str(i_iter+1))
        plots.append(wandb.Image(plt, caption="matplotlib image"))                
        # plt.show()

    ###
    # to plot the information about the reference trajectory and the real trajectory
    ###
    # TODO: not exactly the same starting point
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

    if_plot = 1
    if_both = 0
    if_joint = 0
    if_cylinder = 0

    if if_plot and (i_iter+1)%50==0:
        legend_position = 'best'
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
        # plt.show()

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

            plt.suptitle('Cylinder Space Trajectory Tracking Performance'+' Iter '+str(i_iter+1))
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

            plt.suptitle('Joint Space Real Data'+' Iter '+str(i_iter+1))                
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

            plt.suptitle('Cylinder Space Desired Data'+' Iter '+str(i_iter+1))
            plots.append(wandb.Image(plt, caption="matplotlib image"))                
            # plt.show()
        
        wandb.log({'related plots': plots})

    print('begin {}. optimization'.format(i_iter+1))
    W_list = [None] * num_nn
    for i in range(num_nn):    
        W = []
        [W.append(param.data.view(-1)) for param in cnn_list[i].parameters()]
        W = torch.cat(W)
        W_list[i] = W.cpu().numpy().reshape(-1, 1)

    part4 = [] # num_nn
    for dof in range(num_nn):
        temp = np.zeros((Pamy.y_desired.shape[1],Pamy.y_desired.shape[1]))
        if num_nn==2:
            for row in range(1,Pamy.y_desired.shape[1]):
                temp[row, row-1] = Pamy.pid_for_tracking[dof+1,0]+Pamy.pid_for_tracking[dof+1,2]*100
                if row>1:
                    temp[row, row-2] = -Pamy.pid_for_tracking[dof+1,2]*100
        elif num_nn==3:
            for row in range(1,Pamy.y_desired.shape[1]):
                temp[row, row-1] = Pamy.pid_for_tracking[dof,0]+Pamy.pid_for_tracking[dof,2]*100
                if row>1:
                    temp[row, row-2] = -Pamy.pid_for_tracking[dof,2]*100
        part4.append(temp)
    if method_updating_traj==1:
        part3 = get_grads_list(get_dataset(Pamy.y_desired), cnn_list) # num_nn
    elif method_updating_traj==2:
        part3 = get_grads_list(get_dataset(Pamy.y_desired, option=aug_ref_traj, ref=update_point_index_list), cnn_list) # num_nn
    part2 = [Pamy.O_list[i].Bu for i in PAMY_CONFIG.dof_list] # 4
    part1_temp = (y-theta_)/math.pi*180
    part1 = [part1_temp[i].reshape(1,-1) for i in range(len(part1_temp))] # 4
    loss = [np.linalg.norm(part1_temp[i].reshape(1,-1)) for i in range(len(part1_temp))]

    ###
    # to decay the learning rate
    ###
    # if (i_iter+1)%500==0:
    #     learning_rate *= 0.5

    for dof in range(num_nn):
        if num_nn==2:
            delta = (learning_rate[dof+1]*part1[dof+1]@np.linalg.pinv(np.eye(Pamy.y_desired.shape[1])+part2[dof+1]@part4[dof])@part2[dof+1]@part3[dof]).reshape(-1, 1)
        elif num_nn==3:
            delta = (learning_rate[dof]*part1[dof]@np.linalg.pinv(np.eye(Pamy.y_desired.shape[1])+part2[dof]@part4[dof])@part2[dof]@part3[dof]).reshape(-1, 1)
        W_list[dof] = W_list[dof] - delta
    
    cnn_list = set_parameters(W_list, cnn_list, idx_list, shape_list)
    print('end {}. optimization'.format(i_iter+1))

    if (i_iter+1)%100==0:
        root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(save_path_num) +'/linear_model' + '/' + str(i_iter+1)
        mkdir(root_model_epoch) 
        for dof in range(num_nn):
            cnn = cnn_list[dof]
            root_file = root_model_epoch + '/' + str(dof)
            torch.save(cnn.state_dict(), root_file)

    print(t_used)
    print('inference time consumption: ')
    print(t_inf/(i_iter+1))
    print('loss:')
    print(loss)
    if flag_wandb:
        wandb.log({'loss_0': loss[0], 'loss_1': loss[1], 'loss_2': loss[2]}, i_iter+1)

    i_iter += 1

    if flag_time_record:
        t_end = time.time()
        print('used time: {}'.format(t_end-t_begin))

    if flag_time_analysis:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
