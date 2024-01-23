'''
This script is used to train the robot for better tracking performance with online convex optimization.
control diagram description: trainable feedforward block and fixed feedback block
ff control policy description: CNN + FCN + online gradient descent
fb control policy description: fixed PD controller (PID controller for doing AngleInitialization)
training data description: multiple reference trajectories at the same initial state, with updates 
in a delay of h/f to mimic possible online replanning
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
filter_size     = 3                     # the kernel size for height dimension in CNN
learning_rate   = [5e-2, 5e-2, 5e-2]    # learning rates

# %% initialize the gpu and the robot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    factor = [1.0, 1.0, 1.0]
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
            u[dof, :] = cnn(dataset[0]).cpu().detach().numpy().flatten() * denorm[dof]
        except:
            u[dof, :] = cnn(dataset[0].float()).cpu().detach().numpy().flatten() * denorm[dof]
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
        nn.init.uniform_(l.weight,a=-0.01,b=0.01)
        nn.init.constant_(l.bias,0.0)

for dof in range(3):
    cnn = CNN(channel_in=nr_channel, filter_size=filter_size, height=height, width=width)
    cnn.apply(weight_init)
    cnn.to(device)
    cnn_list.append(cnn)

for name, param in cnn.named_parameters():
    name_list.append(name)
    shape_list.append(param.shape)
    d_idx = len(param.data.view(-1))
    idx += d_idx
    idx_list.append(idx)

print('the number of trainable parameters: {}'.format(idx_list[-1]))

# %% do the online learning
'''
Below codes are used to help with the visualization of the target point 'p_int' and the planned trajectory 'theta'.
'''
# hit_point = handle.frontends["hit_point"]
# hit_point.add_command(p_int.reshape(-1).tolist(),(0,0,0),o80.Duration_us.seconds(1),o80.Mode.QUEUE)
# hit_point.pulse_and_wait()
# for j in range(len(t_stamp)):
#     joints = theta[:,j].reshape(-1).tolist()
#     frontend.add_command(joints,(0,0,0,0),o80.Duration_us.milliseconds(100),o80.Mode.QUEUE)
#     frontend.pulse_and_wait()

wandb.init(
    entity='jubilantrou',
    project='pamy_oco_trial'
)

for i_iter in range(nr_iteration):
    print('------------')
    print('iter {}'.format(i_iter))

    (t, angle) = get_random()
    (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, time_update_record) = RG.updatedPathPlanning(time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle)
    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = np.copy(theta)
    theta = theta - theta[:, 0].reshape(-1, 1)
    Pamy.ImportTrajectory(theta, t_stamp)

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    angle_initial_read = np.array(frontend.latest().get_positions())
    # print('initial joint angles (degree): {}'.format(angle_initial_read/math.pi*180))
    Pamy.GetOptimizer_convex(angle_initial=angle_initial_read, h=h, nr_channel=nr_channel, coupling=coupling)
    u = get_prediction(cnn_list, Pamy.y_desired, PAMY_CONFIG.pressure_limit)

    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list=u, mode_name='ff+fb', coupling=coupling, learning_mode='u')
    y_out = y - y[:, 0].reshape(-1, 1)

    '''
    Below codes are used to plot the information about the reference trajectory and the real trajectory.
    '''
    # TODO: not exactly the same starting point
    # theta_ starts with pos_init while y starts without
    # have added, but still to check the starting point choice and the pressure given style

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
    plots = []

    if if_plot:
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
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Joint Space Trajectory Tracking Performance'+' Iter '+str(i_iter))
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

            plt.suptitle('Cylinder Space Trajectory Tracking Performance'+' Iter '+str(i_iter))
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

            plt.suptitle('Joint Space Real Data'+' Iter '+str(i_iter))                
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

            plt.suptitle('Cylinder Space Desired Data'+' Iter '+str(i_iter))
            plots.append(wandb.Image(plt, caption="matplotlib image"))                
            # plt.show()
        
        wandb.log({'traj related plots': plots})

    print('begin {}. optimization'.format(i_iter))
    W_list = [None] * 3
    for i in range(3):    
        W = []
        [W.append(param.data.view(-1)) for param in cnn_list[i].parameters()]
        W = torch.cat(W)
        W_list[i] = W.cpu().numpy().reshape(-1, 1)

    part4 = [np.zeros((Pamy.y_desired.shape[1],Pamy.y_desired.shape[1]))]*3
    for dof in range(3):
        for row in range(1,Pamy.y_desired.shape[1]):
            part4[dof][row, row-1] = Pamy.pid_for_tracking[dof,0]+Pamy.pid_for_tracking[dof,2]*100
            if row>1:
                part4[dof][row, row-2] = -Pamy.pid_for_tracking[dof,2]*100
    part3 = get_grads_list(get_dataset(Pamy.y_desired), cnn_list)
    part2 = [Pamy.O_list[i].Bu for i in PAMY_CONFIG.dof_list]
    part1_temp = y-theta_
    part1 = [part1_temp[i].reshape(1,-1) for i in range(len(part1_temp))]
    loss = [np.linalg.norm(part1_temp[i].reshape(1,-1)) for i in range(len(part1_temp))]

    for dof in range(3):
        W_list[dof] = W_list[dof] - (learning_rate[dof]*PAMY_CONFIG.pressure_limit[dof]*part1[dof]@np.linalg.pinv(np.eye(Pamy.y_desired.shape[1])+part2[dof]@part4[dof])@part2[dof]@part3[dof]).reshape(-1, 1)
    
    cnn_list = set_parameters(W_list, cnn_list, idx_list, shape_list)
    print('end {}. optimization'.format(i_iter))

    if (i_iter+1)%20==0:
        root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp' + '/' + str(i_iter+1)
        mkdir(root_model_epoch) 
        for dof in range(3):
            cnn = cnn_list[dof]
            root_file = root_model_epoch + '/' + str(dof)
            torch.save(cnn.state_dict(), root_file)

    print('loss:')
    print(loss)
    wandb.log({'loss_0': loss[0], 'loss_1': loss[1], 'loss_2': loss[2]}, i_iter+1)

# index_used = [] 
# root_model_epoch = root_model + '/' + str(i_epoch)  # save the model at each epoch
# mkdir(root_model_epoch)
# wandb.finish() 
# for dof in range(3):
#     cnn = cnn_list[dof]
#     root_file = '/home/mtian/training_log_temp/model' + str(i_epoch) + str(dof)
#     mkdir(root_file)
#     torch.save(cnn.state_dict(), root_file)

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