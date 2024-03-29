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
import time
from RealRobotGeometry import RobotGeometry
import torch.nn as nn
import wandb
from pyinstrument import Profiler
from OCO_paras import get_paras
from OCO_funcs import *
from OCO_plots import *

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

# %% temp var
fb_input = 20

# %% define trainable blocks
print('-'*30)
print('feedforward blocks')
print('-'*30)
if paras.pamy_system=='MIMO':
    block_list, shape_list, idx_list, W_list = trainable_block_init(nn_type=paras.nn_type, nr_channel=paras.nr_channel, 
                                                       height=paras.height, width=paras.width, device=device, hidden_size=paras.hidden_size, model='/home/mtian/Desktop/MPI-intern/training_log_temp_4000/linear_model_ff/300/0')
else:
    block_list, shape_list, idx_list, W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type=paras.nn_type, nr_channel=paras.nr_channel, 
                                                       height=paras.height, width=paras.width, filter_size=paras.filter_size, device=device, hidden_size=paras.hidden_size, model = paras.ff_model)
# print('-'*30)
# print('feedback blocks')
# print('-'*30)
# # TODO: add the parameters of trainable fb blocks into get_paras()
# fb_block_list, fb_shape_list, fb_idx_list, fb_W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type='FCN', nr_channel=1, 
#                                                                 height=fb_input, width=1, filter_size=None, device=device, hidden_size=[16])

# %% do the online learning
if paras.flag_wandb:
    wandb.init(
        entity='jubilantrou',
        project='pamy_oco_trial',
        config = paras
    )

### initialize the hessian part for the Newton method
# TODO: one more for fb part
if paras.method_updating_policy == 'NM':
    hessian_temp = []
    for dof in paras.flag_dof:
        if not dof:
            hessian_temp.append(None)
        else:
            hessian_temp.append(0)         

i_iter = 0

# ### get the reference trajectory
# (t, angle) = get_random()
# # (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
# #     time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method=paras.method_updating_traj)
# (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle, part=0)
# # aug_ref_traj = []
# # for j in range(len(update_point_index_list)):
# #     comp = get_compensated_data(data=np.hstack((theta[:,:update_point_index_list[j]], theta_list[j][:,time_update_record[j]:])), h_l=paras.h_l, h_r=0)
# #     comp = comp - comp[:,0].reshape(-1, 1) # rel
# #     aug_ref_traj.append(comp)
# theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
# theta_ = np.copy(theta) # absolute values of the reference
# theta = theta - theta[:, 0].reshape(-1, 1) # relative values of the reference
# Pamy.ImportTrajectory(theta, t_stamp)

# # aug_ref_traj.append(get_compensated_data(data=Pamy.y_desired, h_l=paras.h_l, h_r=paras.h_r))

while True:
    if paras.flag_time_analysis:
        profiler = Profiler()
        profiler.start()

    print('-'*30)
    print('iter {}'.format(i_iter+1))
    print('-'*30)

    ### get the reference trajectory
    (t, angle) = get_random()
    (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
        time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method=paras.method_updating_traj)
    aug_ref_traj = []
    for j in range(len(update_point_index_list)):
        comp = get_compensated_data(data=np.hstack((theta[:,:update_point_index_list[j]], theta_list[j][:,time_update_record[j]:])), h_l=paras.h_l, h_r=0)
        comp = comp - comp[:,0].reshape(-1, 1) # rel
        aug_ref_traj.append(comp)
    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = np.copy(theta) # absolute values of the reference
    theta = theta - theta[:, 0].reshape(-1, 1) # relative values of the reference
    Pamy.ImportTrajectory(theta, t_stamp)

    ### get the identified approximate linear model
    Pamy.GetOptimizer_convex(angle_initial=PAMY_CONFIG.GLOBAL_INITIAL, nr_channel=paras.nr_channel, coupling=paras.coupling)   

    ### get the feedforward inputs
    aug_ref_traj.append(get_compensated_data(data=Pamy.y_desired, h_l=paras.h_l, h_r=paras.h_r))
    datapoint = get_datapoints_pro(aug_data=aug_ref_traj, ref=update_point_index_list, window_size=(paras.h_l+paras.h_r+1), ds=paras.ds, device=device, nn_type=paras.nn_type)
    # datapoint = get_datapoints(aug_data=get_compensated_data(data=Pamy.y_desired, h_l=paras.h_l, h_r=paras.h_r), window_size=(paras.h_l+paras.h_r+1), ds=paras.ds, device=device, nn_type=paras.nn_type)
    u = get_prediction(datapoints=datapoint, block_list=block_list)

    ### get the real output
    # TODO
    # Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    # Pamy.PressureInitialization(duration=1)
    (t, step, position, diff, theta_zero) = Pamy.LQRTesting(amp = np.array([[30], [30], [30]])/180*math.pi, t_start = 0.0, t_duration = 6.0)

    # Pamy.PressureInitialization(duration=1)
    # reset_pressures = np.array(Pamy.frontend.latest().get_observed_pressures())
    # Pamy.anchor_ago_list = reset_pressures[:, 0]
    # Pamy.anchor_ant_list = reset_pressures[:, 1]
    # print('reset anchor pressures:')
    # print(reset_pressures)

    angle_initial_read = np.array(frontend.latest().get_positions())
    (y, ff, fb, obs_ago, obs_ant, des_ago, des_ant, fb_datasets) = Pamy.online_convex_optimization(b_list=u, mode_name='ff+fb', coupling=paras.coupling, learning_mode='u', trainable_fb=None, device=device, dim_fb=fb_input)
    y_out = y - y[:, 0].reshape(-1, 1) # relative values of the real output

    print('ranges of feedforward inputs:')
    print('dof 0: {} ~ {}'.format(min(ff[0,:]),max(ff[0,:])))
    print('dof 1: {} ~ {}'.format(min(ff[1,:]),max(ff[1,:])))
    print('dof 2: {} ~ {}'.format(min(ff[2,:]),max(ff[2,:])))
    print('ranges of feedback inputs:')
    print('dof 0: {} ~ {}'.format(min(fb[0,:]),max(fb[0,:])))
    print('dof 1: {} ~ {}'.format(min(fb[1,:]),max(fb[1,:])))
    print('dof 2: {} ~ {}'.format(min(fb[2,:]),max(fb[2,:])))

    # SI_ref = (Pamy.O_list[0].Bu @ ((u.T.reshape(1,-1)).T)).reshape(-1)
    SI_ref = [Pamy.O_list[i].Bu@(ff[i].reshape(-1,1)) for i in range(3)]
    if paras.flag_wandb:
        wandb_plot(i_iter=i_iter, frequency=1, t_stamp=t_stamp, ff=ff, fb=fb, y=y, theta_=theta_, t_stamp_list=t_stamp_list, theta_list=theta_list, T_go_list=T_go_list, p_int_record=p_int_record, 
                   obs_ago=obs_ago, obs_ant=obs_ant, des_ago=des_ago, des_ant=des_ant)
        # wandb_plot(i_iter=i_iter, frequency=1, t_stamp=t_stamp, ff=ff, fb=fb, y=y, theta_=theta_, t_stamp_list=[], theta_list=[], T_go_list=[], p_int_record=[], 
        #            obs_ago=obs_ago, obs_ant=obs_ant, des_ago=des_ago, des_ant=des_ant, SI_ref = None)

    ### compute gradients that will be used to update parameters 
    # 3*Nt x nff
    par_paiff_par_wff = get_par_pai_par_w_list(datapoints=datapoint, block_list=block_list)

    # Nt*nfb
    # par_paifb_par_wfb, par_paifb_par_fbin = get_grads_list(dataset=fb_datasets, block_list=fb_block_list, additional=True)

    ### learnable PD update
    # def get_par_paifb_par_y(flag_dof, dim, par_paifb_par_fbin, dim_fb):
    #     par_paifb_par_y = []
    #     choosing_matrix = np.concatenate((np.zeros((dim_fb,dim)), np.eye(dim_fb), np.zeros((dim_fb,dim+1))), axis=1)
    #     for dof,flag in enumerate(flag_dof):
    #         if not flag:
    #             par_paifb_par_y.append(None)
    #         else:
    #             temp = np.zeros((dim,dim))
    #             for i in range(dim):
    #                 temp[i,:] = par_paifb_par_fbin[dof][i,:].reshape(1,-1)@choosing_matrix[:,-1-i-dim:-1-i]
    #             par_paifb_par_y.append(temp)
    #     return par_paifb_par_y
    # # Nt*Nt          
    # par_paifb_par_y = get_par_paifb_par_y(flag_dof=paras.flag_dof, dim=Pamy.y_desired.shape[1], par_paifb_par_fbin=par_paifb_par_fbin, dim_fb=fb_input)

    ### fixed PD update
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
    # 3*Nt x 3*Nt            
    par_paifb_par_y = get_par_paifb_par_y(flag_dof=paras.flag_dof, dim=Pamy.y_desired.shape[1], pid=Pamy.pid_for_tracking)
    # par_paifb_par_y = [np.zeros((3*Pamy.y_desired.shape[1], 3*Pamy.y_desired.shape[1]))]

    # 3*Nt x 3*Nt
    # TODO
    par_G_par_u = [Pamy.O_list[i].Bu for i in PAMY_CONFIG.dof_list][:3]
    # 1 x 3*Nt
    # TODO: need chenges for MIMO and for adding weights
    par_l_par_y = [((y_out-theta)[i]).reshape(1,-1) for i in range(3)]
    # idx_amp = [1,10,4]
    # for idx_ly in [1,2]:
    #     par_l_par_y[idx_ly][0,50:150] *= idx_amp[idx_ly]
    # 3*Nt x nff
    par_y_par_wff = []
    if paras.pamy_system=='MIMO':
        temp = np.linalg.pinv(np.eye(3*Pamy.y_desired.shape[1])+par_G_par_u[0]@par_paifb_par_y[0])@par_G_par_u[0]@par_paiff_par_wff[0]
        par_y_par_wff.append(temp)
    else:
        for dof,flag in enumerate(paras.flag_dof):
            if not flag:
                par_y_par_wff.append(None)
            else:
                temp = np.linalg.pinv(np.eye(Pamy.y_desired.shape[1])+par_G_par_u[dof]@par_paifb_par_y[dof])@par_G_par_u[dof]@par_paiff_par_wff[dof]
                par_y_par_wff.append(temp)
    # Nt*nfb
    # par_y_par_wfb = []
    # for dof,flag in enumerate(paras.flag_dof):
    #     if not flag:
    #         par_y_par_wfb.append(None)
    #     else:
    #         temp = np.linalg.pinv(np.eye(Pamy.y_desired.shape[1])+par_G_par_u[dof]@par_paifb_par_y[dof])@par_G_par_u[dof]@par_paifb_par_wfb[dof]
    #         par_y_par_wfb.append(temp)

    def get_new_parameters(W_list, method, lr, par_l_par_y, par_y_par_w, par_pai_par_w):
        for dof in range(len(par_l_par_y)):
            if par_y_par_w[dof] is None:
                W_list[dof] = None
                continue
            if method == 'GD':
                # TODO: first ele in list of lr works for MIMO
                delta = (lr[dof]*par_l_par_y[dof]@par_y_par_w[dof]).reshape(-1, 1)
            elif method == 'NM':
                hessian_temp[dof] += par_y_par_w[dof].T@par_y_par_w[dof]+paras.alpha_list[dof]*par_pai_par_w[dof].T@par_pai_par_w[dof]+paras.epsilon_list[dof]*np.eye(par_pai_par_w[dof].shape[1])
                # hessian_temp[dof] = par_y_par_w[dof].T@par_y_par_w[dof]+paras.alpha_list[dof]*par_pai_par_w[dof].T@par_pai_par_w[dof]+0.0*np.eye(par_pai_par_w[dof].shape[1])
                delta = (lr[dof]*np.linalg.pinv(hessian_temp[dof]/(i_iter+1))@par_y_par_w[dof].T@par_l_par_y[dof].T).reshape(-1 ,1)
            W_list[dof] = W_list[dof] - delta
        return W_list, delta  
      
    W_list, delta = get_new_parameters(W_list=W_list, method=paras.method_updating_policy, lr=paras.lr_list, par_l_par_y=par_l_par_y, par_y_par_w=par_y_par_wff, par_pai_par_w=par_paiff_par_wff)   
    block_list = set_parameters(W_list, block_list, idx_list, shape_list, device)
    # fb_W_list, fb_delta = get_new_parameters(W_list=fb_W_list, method=paras.method_updating_policy, lr=[5.0, 5.0, 5.0], par_l_par_y=par_l_par_y, par_y_par_w=par_y_par_wfb, par_pai_par_w=par_paifb_par_wfb)   
    # fb_block_list = set_parameters(fb_W_list, fb_block_list, fb_idx_list, fb_shape_list, device)

    if (i_iter+1)%50==0:
        root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(paras.save_path_num) +'/linear_model_ff' + '/' + str(i_iter+1)
        mkdir(root_model_epoch) 
        for dof,cnn in enumerate(block_list):
            if cnn is None:
                continue
            root_file = root_model_epoch + '/' + str(dof)
            torch.save(cnn.state_dict(), root_file)
        # fb_root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(paras.save_path_num) +'/linear_model_fb' + '/' + str(i_iter+1)
        # mkdir(fb_root_model_epoch) 
        # for dof,cnn in enumerate(fb_block_list):
        #     if cnn is None:
        #         continue
        #     fb_root_file = fb_root_model_epoch + '/' + str(dof)
        #     torch.save(cnn.state_dict(), fb_root_file)
    
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
        wandb.log({'loss_0': loss[0]/t_stamp[-1]/math.pi*180/100, 'loss_1': loss[1]/t_stamp[-1]/math.pi*180/100, 'loss_2': loss[2]/t_stamp[-1]/math.pi*180/100}, i_iter+1)

    i_iter += 1

    if paras.flag_time_analysis:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
