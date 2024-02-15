'''
This script is used to train the robot for better tracking performance with online convex optimization.
* [control diagram description] trainable feedforward block and fixed feedback block
* [ff control policy description] (CNN +) FCN + online gradient descent/online Newton method
* [fb control policy description] fixed PD controller (another PID controller to do AngleInitialization)
* [training data description] multiple reference trajectories, at the same initial state and with mimic updates
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
from OCO_paras import get_paras
from OCO_funcs import *

# %% initialize the parameters, the gpu and the robot
paras = get_paras()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))
fix_seed(paras.seed)

if paras.obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif paras.obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
Pamy = PAMY_CONFIG.build_pamy(frontend=frontend, obj=paras.obj)
RG = RobotGeometry()

# %% create functions (TODO: to organize often used ones as an independent script)
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)    

# %% define trainable blocks
def weight_init(layer):
    '''
    to initialize the weights in your own way
    '''
    if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def trainable_blocks_init(paras):
    '''
    to initialize the trainable policy blocks
    '''
    cnn_list   = []
    name_list  = []
    shape_list = []
    idx_list   = []
    idx = 0
    idx_list.append(idx)

    for index, dof in enumerate(paras.flag_dof):
        if not dof:
            cnn_list.append(None)
        else:
            cnn = CNN(channel_in=paras.nr_channel, filter_size=paras.filter_size, height=paras.height, width=paras.width)
            ### for pre-trained weights loading
            temp_model = '/home/mtian/Desktop/MPI-intern/training_log_temp_8/linear_model/5200/' + str(index)
            temp = torch.load(temp_model)
            cnn.load_state_dict(temp)
            ### for self-defined weights initialization
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
    print(len(shape_list))

    return cnn_list, shape_list, idx_list

cnn_list, shape_list, idx_list = trainable_blocks_init(paras)

# %% do the online learning
if paras.flag_wandb:
    wandb.init(
        entity='jubilantrou',
        project='pamy_oco_trial'
    )

i_iter = 0
if paras.method_updating_policy == 'NM':
    hessian_temp = []
    for dof in paras.flag_dof:
        if not dof:
            hessian_temp.append(None)
        else:
            hessian_temp.append(0)
W_list = []
for cnn in cnn_list:
    if cnn is None:
        W_list.append(None)
    else:    
        W = []
        [W.append(param.data.view(-1)) for param in cnn.parameters()]
        W = torch.cat(W)
        W_list.append(W.cpu().numpy().reshape(-1, 1))

# for i in range(1):
#     (t, angle) = get_random()
# (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
#     time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method=paras.method_updating_traj)
# aug_ref_traj = []
# for j in range(len(update_point_index_list)):
#     comp = get_compensated_data(data=np.hstack((theta[:,:update_point_index_list[j]], theta_list[j][:,time_update_record[j]:])), h_l=paras.h_l, h_r=paras.h_r, option='only_left')
#     comp = comp - comp[:,0].reshape(-1, 1) # rel
#     aug_ref_traj.append(comp)

# theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
# theta_ = np.copy(theta)
# theta = theta - theta[:, 0].reshape(-1, 1) # rel
# Pamy.ImportTrajectory(theta, t_stamp)

### for penalty parameters testing
# rec = 0

# fig1 = plt.figure(figsize=(18, 18))
# ax1_position0 = fig1.add_subplot(111)
# plt.xlabel(r'Time $t$ in s')
# plt.ylabel(r'Dof_0')
# line = []

# for i in range(200):
#     (t, angle) = get_random()
#     (p, v, a, j, theta, t_stamp) = RG.PathPlanning(
#         time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, part=0)
    
#     line_temp, = ax1_position0.plot(t_stamp, theta[0, :]/math.pi*180, linewidth=1)
#     line.append( line_temp )

# line_temp, = ax1_position0.plot(t_stamp, [120]*len(t_stamp), linewidth=1)
# line.append( line_temp )
# line_temp, = ax1_position0.plot(t_stamp, [-120]*len(t_stamp), linewidth=1)
# line.append( line_temp )

# plt.legend(handles=line, shadow=True)
# plt.show()
###


log_min = []
log_max = []
log_1 = []
log_2 = []

while True:
    if paras.flag_time_analysis:
        profiler = Profiler()
        profiler.start()

    print('-'*30)
    print('iter {}'.format(i_iter+1))
    print('-'*30)

    (t, angle) = get_random()
    (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
        time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method=paras.method_updating_traj)
    aug_ref_traj = []
    for j in range(len(update_point_index_list)):
        comp = get_compensated_data(data=np.hstack((theta[:,:update_point_index_list[j]], theta_list[j][:,time_update_record[j]:])), h_l=paras.h_l, h_r=paras.h_r, option='only_left')
        comp = comp - comp[:,0].reshape(-1, 1) # rel
        aug_ref_traj.append(comp)

    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = np.copy(theta)
    theta = theta - theta[:, 0].reshape(-1, 1) # rel
    Pamy.ImportTrajectory(theta, t_stamp)

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    angle_initial_read = np.array(frontend.latest().get_positions())
    Pamy.GetOptimizer_convex(angle_initial=angle_initial_read, nr_channel=paras.nr_channel, coupling=paras.coupling)
    datapoint = get_datapoint(y=Pamy.y_desired, h_l=paras.h_l, h_r=paras.h_r, ds=paras.ds, device=device, sub_traj=aug_ref_traj, ref=update_point_index_list)
    # datapoint = get_datapoint(y=Pamy.y_desired, h_l=paras.h_l, h_r=paras.h_r, ds=paras.ds, device=device)
    u = get_prediction(datapoint=datapoint, cnn_list=cnn_list, y=Pamy.y_desired)

    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list=u, mode_name='ff+fb', coupling=paras.coupling, learning_mode='u')
    # y[2,:] = math.pi/2 - y[2,:]
    y_out = y - y[:, 0].reshape(-1, 1) #rel

    print('pressure check:')
    print('dof 0: {} ~ {}'.format(min(ff[0,:]),max(ff[0,:])))
    print('dof 1: {} ~ {}'.format(min(ff[1,:]),max(ff[1,:])))
    print('dof 2: {} ~ {}'.format(min(ff[2,:]),max(ff[2,:])))

    if paras.flag_wandb:
        wandb_plot(i_iter=i_iter, frequency=10, t_stamp=t_stamp, ff=ff, fb=fb, y=y, theta_=theta_, t_stamp_list=t_stamp_list, theta_list=theta_list, T_go_list=T_go_list, p_int_record=p_int_record)

    def get_par_paifb_par_y(flag_dof, dim, pid):
        par_paifb_par_y = []
        for dof,flag in enumerate(flag_dof):
            if not flag:
                par_paifb_par_y.append(None)
            else:
                temp = np.zeros((dim,dim))
                for row in range(1,dim):
                    temp[row, row-1] = pid[dof,0]+pid[dof,2]*100
                    if row>1:
                        temp[row, row-2] = -pid[dof,2]*100
                par_paifb_par_y.append(temp)
        return par_paifb_par_y
    '''
    Nt x Nt
    '''            
    par_paifb_par_y = get_par_paifb_par_y(flag_dof=paras.flag_dof, dim=Pamy.y_desired.shape[1], pid=Pamy.pid_for_tracking)

    '''
    Nt x nff
    '''
    par_paiff_par_wff = get_grads_list(dataset=get_dataset(datapoint=datapoint, batch_size=1), cnn_list=cnn_list)
    
    '''
    Nt x Nt
    ''' 
    par_G_par_u = [Pamy.O_list[i].Bu for i in PAMY_CONFIG.dof_list][:3]

    '''
    1 x Nt
    ''' 
    par_l_par_y = [(y-theta_)[i].reshape(1,-1) for i in range(len(y))][:3]

    '''
    Nt x nff
    '''
    par_y_par_wff = []
    for dof,flag in enumerate(paras.flag_dof):
        if not flag:
            par_y_par_wff.append(None)
        else:
            temp = np.linalg.pinv(np.eye(Pamy.y_desired.shape[1])+par_G_par_u[dof]@par_paifb_par_y[dof])@par_G_par_u[dof]@par_paiff_par_wff[dof]
            par_y_par_wff.append(temp)

    def get_new_parameters(W_list, method, lr, par_l_par_y, par_y_par_wff, par_paiff_par_wff):
        for dof in range(len(par_l_par_y)):
            if par_y_par_wff[dof] is None:
                W_list[dof] = None
                continue
            if method == 'GD':
                delta = (lr[dof]*par_l_par_y[dof]@par_y_par_wff[dof]).reshape(-1, 1)
            elif method == 'NM':
                hessian_temp[dof] += par_y_par_wff[dof].T@par_y_par_wff[dof]+paras.alpha_list[dof]*par_paiff_par_wff[dof].T@par_paiff_par_wff[dof]+paras.epsilon_list[dof]*np.eye(par_paiff_par_wff[dof].shape[1])
                delta = (lr[dof]*np.linalg.pinv(hessian_temp[dof]/(i_iter+1))@par_y_par_wff[dof].T@par_l_par_y[dof].T).reshape(-1 ,1)
            W_list[dof] = W_list[dof] - delta
        return W_list, delta
    
    W_list, delta = get_new_parameters(W_list=W_list, method=paras.method_updating_policy, lr=paras.lr_list, par_l_par_y=par_l_par_y, par_y_par_wff=par_y_par_wff, par_paiff_par_wff=par_paiff_par_wff)
    log_min.append(min(delta))
    log_max.append(max(delta))
    log_1.append(par_l_par_y[2])
    log_2.append(par_y_par_wff[2])
    
    cnn_list = set_parameters(W_list, cnn_list, idx_list, shape_list, device)

    # if (i_iter+1)>30:
    #     time.sleep(0.1)

    if (i_iter+1)%100==0:
        root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(paras.save_path_num) +'/linear_model' + '/' + str(i_iter+1)
        mkdir(root_model_epoch) 
        for dof,cnn in enumerate(cnn_list):
            if cnn is None:
                continue
            root_file = root_model_epoch + '/' + str(dof)
            torch.save(cnn.state_dict(), root_file)
    
    if (i_iter+1)%20==0:
        root_file = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(paras.save_path_num) +'/linear_model/log_data/' + str(i_iter+1)
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

    part1_temp = (y-theta_)
    loss = [np.linalg.norm(part1_temp[i].reshape(1,-1)) for i in range(len(part1_temp))]
    print('loss:')
    print(loss)
    if paras.flag_wandb:
        wandb.log({'loss_0': loss[0]/t_stamp[-1]/math.pi*180, 'loss_1': loss[1]/t_stamp[-1]/math.pi*180, 'loss_2': loss[2]/t_stamp[-1]/math.pi*180}, i_iter+1)

    i_iter += 1

    if paras.flag_time_analysis:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
