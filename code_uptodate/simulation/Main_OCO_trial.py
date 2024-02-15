'''
This script is used to train the robot for better tracking performance with online convex optimization.
* [control diagram] trainable feedforward block + fixed/trainable feedback block
* [ff control policy] FCN; CNN + FCN
* [fb control policy] fixed PD controller (another PID controller for AngleInitialization); FCN
* [training data] multiple manually generated reference trajectories, at the same initial state and with mimic updates
* [training method] online gradient descent/online Newton method
'''
# %% import libraries
import PAMY_CONFIG
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
from get_handle import get_handle
import o80_pam
import torch
from Trainable_blocks import *
import torch.nn as nn
import time
from RealRobotGeometry import RobotGeometry
import torch.nn as nn
import wandb
from pyinstrument import Profiler
from OCO_paras import get_paras
from OCO_funcs import *

# %% initialize the parameters, the gpu and the robot
paras = get_paras()
PAMY_CONFIG.obj = paras.obj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))
fix_seed(seed=paras.seed)

if paras.obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif paras.obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
RG = RobotGeometry(initial_posture=PAMY_CONFIG.GLOBAL_INITIAL)   

# %% define trainable blocks
print('-'*30)
print('feedforward blocks')
print('-'*30)
block_list, shape_list, idx_list, W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type=paras.nn_type, nr_channel=paras.nr_channel, 
                                                       height=paras.height, width=paras.width, filter_size=paras.filter_size, device=device, hidden_size=paras.hidden_size)
print('-'*30)
print('feedback blocks')
print('-'*30)
# TODO: add the parameters of trainable fb blocks into get_paras()
fb_block_list, fb_shape_list, fb_idx_list, fb_W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type='FCN', nr_channel=1, 
                                                                height=8, width=1, filter_size=None, device=device, hidden_size=[16])

# %% do the online learning
if paras.flag_wandb:
    wandb.init(
        entity='jubilantrou',
        project='pamy_oco_trial'
    )

if paras.method_updating_policy == 'NM':
    hessian_temp = []
    for dof in paras.flag_dof:
        if not dof:
            hessian_temp.append(None)
        else:
            hessian_temp.append(0)

i_iter = 0
while True:
    if paras.flag_time_analysis:
        profiler = Profiler()
        profiler.start()

    print('-'*30)
    print('iter {}'.format(i_iter+1))
    print('-'*30)

    (t, angle) = get_random()
    # TODO: review from here
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
    u = get_prediction(datapoint=datapoint, cnn_list=block_list, y=Pamy.y_desired)

    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list=u, mode_name='ff+fb', coupling=paras.coupling, learning_mode='u')
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
    par_paiff_par_wff = get_grads_list(dataset=get_dataset(datapoint=datapoint, batch_size=1), cnn_list=block_list)
    
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
    
    block_list = set_parameters(W_list, block_list, idx_list, shape_list, device)

    # if (i_iter+1)>30:
    #     time.sleep(0.1)

    if (i_iter+1)%100==0:
        root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(paras.save_path_num) +'/linear_model' + '/' + str(i_iter+1)
        mkdir(root_model_epoch) 
        for dof,cnn in enumerate(block_list):
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
