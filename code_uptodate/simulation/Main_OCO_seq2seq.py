# %% import libraries
import PAMY_CONFIG
import math
import numpy as np
import pickle5 as pickle
from get_handle import get_handle
import o80_pam
import torch
from Trainable_blocks import *
from RealRobotGeometry import RobotGeometry
import wandb
from pyinstrument import Profiler
from OCO_paras import get_paras
from OCO_utils import *
from OCO_plots import *
import scipy

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
fb_input = 10

# %% define trainable blocks
print('-'*30)
print('feedforward blocks')
print('-'*30)
block_list, shape_list, idx_list, W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type='LinearMap', nr_channel=paras.nr_channel, 
                                                                height=270, width=paras.width, filter_size=paras.filter_size, device=device, hidden_size=[32])

# if paras.pamy_system=='MIMO':
#     block_list, shape_list, idx_list, W_list = trainable_block_init(nn_type=paras.nn_type, nr_channel=paras.nr_channel, 
#                                                        height=paras.height, width=paras.width, device=device, hidden_size=paras.hidden_size, model='/home/mtian/Desktop/MPI-intern/training_log_temp_4000/linear_model_ff/300/0')
# else:
#     block_list, shape_list, idx_list, W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type=paras.nn_type, nr_channel=paras.nr_channel, 
#                                                        height=paras.height, width=paras.width, filter_size=paras.filter_size, device=device, hidden_size=paras.hidden_size, model = paras.ff_model)
#     # disturb_block_list, disturb_shape_list, disturb_idx_list, disturb_W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type=paras.nn_type, nr_channel=paras.nr_channel, 
#     #                                                                             height=21, width=paras.width, filter_size=paras.filter_size, device=device, hidden_size=[32], disturbance=[5e-4,2.5e-4,2e-3], 
#     #                                                                             model='/home/mtian/Desktop/MPI-intern/training_log_temp_12800/linear_model_disturb/100/')
#     disturb_block_list, disturb_shape_list, disturb_idx_list, disturb_W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type=paras.nn_type, nr_channel=paras.nr_channel, 
#                                                                                 height=21, width=paras.width, filter_size=paras.filter_size, device=device, hidden_size=[32])
    
# print('-'*30)
# print('feedback blocks')
# print('-'*30)
# TODO: add the parameters of trainable fb blocks into get_paras()
# fb_block_list, fb_shape_list, fb_idx_list, fb_W_list = trainable_blocks_init(flag_dof=paras.flag_dof, nn_type='FCN', nr_channel=1, 
#                                                                 height=fb_input, width=1, filter_size=None, device=device, hidden_size=[16], linear=True)
# fb_block_list[0] = None

# %% do the online learning
if paras.flag_wandb:
    run = wandb.init(
        entity='jubilantrou',
        project='pamy_oco_trial',
        config = paras
    )

### initialize the hessian part for the Newton method
# TODO: one more for fb part
# TODO: one more for disturbances part
if paras.method_updating_policy == 'NM':
    hessian_temp = []
    fb_hessian_temp = []
    disturb_hessian_temp = []
    for dof in paras.flag_dof:
        if not dof:
            hessian_temp.append(None)
            fb_hessian_temp.append(None)
            disturb_hessian_temp.append(None)
        else:
            hessian_temp.append(0)
            fb_hessian_temp.append(0)
            disturb_hessian_temp.append(0)         

i_iter = 0

# for exp_trained in range(31):
#     (t, angle) = get_random()

### get the fixed trajectory
# (t, angle) = get_random()
# T_go = t
# (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle, part=0)
# theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
# theta_ = np.copy(theta) # absolute values of the reference
# theta = theta - theta[:, 0].reshape(-1, 1) # relative values of the reference
# Pamy.ImportTrajectory(theta, t_stamp)

while True:
    if paras.flag_time_analysis:
        profiler = Profiler()
        profiler.start()

    print('-'*30)
    print('iter {}'.format(i_iter+1))
    print('-'*30)

    ### get the reference trajectory
    # TODO: add an API for fixed trajectory testing
    # (t, angle) = get_random()
    # (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
    #     time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method=paras.method_updating_traj)
    
    # aug_ref_traj = []
    # for j in range(len(update_point_index_list)):
    #     comp = get_compensated_data(data=np.hstack((theta[:,:update_point_index_list[j]], theta_list[j][:,time_update_record[j]:])), h_l=paras.h_l, h_r=0)
    #     comp = comp - comp[:,0].reshape(-1, 1) # rel
    #     aug_ref_traj.append(comp)

    # # disturb_aug_ref_traj = []
    # # for j in range(len(update_point_index_list)):
    # #     comp = get_compensated_data(data=np.hstack((theta[:,:update_point_index_list[j]], theta_list[j][:,time_update_record[j]:])), h_l=10, h_r=0)
    # #     comp = comp - comp[:,0].reshape(-1, 1) # rel
    # #     disturb_aug_ref_traj.append(comp)

    # theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    # theta_ = np.copy(theta) # absolute values of the reference
    # theta = theta - theta[:, 0].reshape(-1, 1) # relative values of the reference
    # Pamy.ImportTrajectory(theta, t_stamp)
    # aug_ref_traj.append(get_compensated_data(data=Pamy.y_desired, h_l=paras.h_l, h_r=paras.h_r))

    (t, angle) = get_random()
    T_go = t
    (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle, part=0)
    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = np.copy(theta) # absolute values of the reference
    theta = theta - theta[:, 0].reshape(-1, 1) # relative values of the reference
    Pamy.ImportTrajectory(theta, t_stamp)

    ### get the identified approximate linear model
    Pamy.GetOptimizer_convex(angle_initial=PAMY_CONFIG.GLOBAL_INITIAL, nr_channel=paras.nr_channel, coupling=paras.coupling)   

    ### get the feedforward inputs
    # TODO: add an API for fixed trajectory testing
    # datapoint = get_datapoints_pro(aug_data=aug_ref_traj, ref=update_point_index_list, window_size=(paras.h_l+paras.h_r+1), ds=paras.ds, device=device, nn_type=paras.nn_type)
    # datapoint = get_datapoints(aug_data=get_compensated_data(data=Pamy.y_desired, h_l=paras.h_l, h_r=paras.h_r), window_size=(paras.h_l+paras.h_r+1), ds=paras.ds, device=device, nn_type=paras.nn_type)
    # u = get_prediction(datapoints=datapoint, block_list=block_list)
    compensated_data_input = get_compensated_data(data=Pamy.y_desired, h_l=0, h_r=(270-Pamy.y_desired.shape[1]))
    # TODO: different process for LinearMap and CNN
    data_input = torch.tensor(compensated_data_input[0:3,:]/np.pi*180, dtype=float).reshape(1, -1).to(device)
    u = np.zeros((3, Pamy.y_desired.shape[1]))
    for dof, block in enumerate(block_list):
        if block is None:
            continue
        else:
            block.eval()
            try:
                u_temp = block(data_input).cpu().detach().numpy()
            except:
                u_temp = block(data_input.float()).cpu().detach().numpy()           
            u[dof, :] = u_temp.flatten()[0:Pamy.y_desired.shape[1]]

    # disturb_aug_ref_traj.append(get_compensated_data(data=Pamy.y_desired, h_l=10, h_r=10))
    # disturb_datapoint = get_datapoints_pro(aug_data=disturb_aug_ref_traj, ref=update_point_index_list, window_size=21, ds=1, device=device, nn_type=paras.nn_type)
    # disturb_datapoint = get_datapoints(aug_data=get_compensated_data(data=Pamy.y_desired, h_l=10, h_r=10), window_size=21, ds=1, device=device, nn_type=paras.nn_type)
    # d = get_prediction(datapoints=disturb_datapoint, block_list=disturb_block_list)

    # 3*Nt x 3*Nt
    # TODO
    par_G_par_u = [Pamy.O_list[i].Bu for i in PAMY_CONFIG.dof_list][:3]
    # Bd_list = [Pamy.O_list[i].Bd for i in PAMY_CONFIG.dof_list][:3]
    # u_add = np.zeros(u.shape)
    # for idx_u_add in range(3):
    #     u_add[idx_u_add, :] = (np.linalg.pinv(par_G_par_u[idx_u_add])@(Bd_list[idx_u_add]@(d[idx_u_add,:].reshape(-1,1)))).reshape(-1)

    ### get the real output
    # TODO
    # Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    # Pamy.PressureInitialization(duration=1)

    print('___for checking___')
    print('currrent max input is: {}'.format(np.max(np.abs(u))))

    (t, step, position, diff, theta_zero) = Pamy.LQRTesting(amp = np.array([[30], [30], [30]])/180*math.pi, t_start = 0.0, t_duration = 6.0)

    # Pamy.PressureInitialization(duration=1)
    # reset_pressures = np.array(Pamy.frontend.latest().get_observed_pressures())
    # Pamy.anchor_ago_list = reset_pressures[:, 0]
    # Pamy.anchor_ant_list = reset_pressures[:, 1]
    # print('reset anchor pressures:')
    # print(reset_pressures)

    angle_initial_read = np.array(frontend.latest().get_positions())
    (y, ff, fb, obs_ago, obs_ant, des_ago, des_ant, fb_datasets) = Pamy.online_convex_optimization(b_list=(u), mode_name='ff+fb', coupling=paras.coupling, learning_mode='u', trainable_fb=None, device=device, dim_fb=fb_input)
    y_out = y - y[:, 0].reshape(-1, 1) # relative values of the real output

    print('ranges of feedforward inputs:')
    print('dof 0: {} ~ {}'.format(min(u[0,:]),max(u[0,:])))
    print('dof 1: {} ~ {}'.format(min(u[1,:]),max(u[1,:])))
    print('dof 2: {} ~ {}'.format(min(u[2,:]),max(u[2,:])))
    print('ranges of feedback inputs:')
    print('dof 0: {} ~ {}'.format(min(fb[0,:]),max(fb[0,:])))
    print('dof 1: {} ~ {}'.format(min(fb[1,:]),max(fb[1,:])))
    print('dof 2: {} ~ {}'.format(min(fb[2,:]),max(fb[2,:])))
    # print('ranges of compensation inputs:')
    # print('dof 0: {} ~ {}'.format(min(d[0,:]),max(d[0,:])))
    # print('dof 1: {} ~ {}'.format(min(d[1,:]),max(d[1,:])))
    # print('dof 2: {} ~ {}'.format(min(d[2,:]),max(d[2,:])))

    (_, end_ref) = RG.AngleToEnd(theta_)
    (_, end_real) = RG.AngleToEnd(y)
    if paras.flag_wandb:
        # plots_to_show = wandb_plot(i_iter=i_iter, frequency=1, t_stamp=t_stamp, ff=u, fb=u_add, y=y, theta_=theta_, t_stamp_list=t_stamp_list, theta_list=theta_list, T_go_list=T_go_list, p_int_record=p_int_record, 
        #            obs_ago=obs_ago, obs_ant=obs_ant, des_ago=des_ago, des_ant=des_ant, disturbance=d, end_ref=end_ref, end_real=end_real)
        plots_to_show = wandb_plot(i_iter=i_iter, frequency=1, t_stamp=t_stamp, ff=ff, fb=fb, y=y, theta_=theta_, t_stamp_list=[], theta_list=[], T_go_list=[T_go], p_int_record=[], 
                   obs_ago=obs_ago, obs_ant=obs_ant, des_ago=des_ago, des_ant=des_ant, disturbance=None, end_ref=end_ref, end_real=end_real)

    ### compute gradients that will be used to update parameters 
    # 3*Nt x nff
    par_paiff_par_wff = []
    for idx, block in enumerate(block_list):
        if block is None:
            par_paiff_par_wff.append(None)
            continue

        block.train()
        for param in block.parameters():
            if param.grad is None:
                break
            param.grad.zero_()

        flag = True
        for idx_out in range(Pamy.y_desired.shape[1]):
            try:
                pai_temp = block(data_input.float())
            except:
                pai_temp = block(data_input)
        
            pai_temp[0, idx_out].backward()
            grad = []
            for param in block.parameters():
                grad.append(torch.clone(param.grad.cpu().view(-1)))
                param.grad.zero_()

            grad_ele = torch.cat(grad)
        
            grads = np.copy(grad_ele.reshape(1, -1)) if flag else np.concatenate((grads, grad_ele.reshape(1, -1)), axis=0)
            flag = False

        assert grads.shape[0] == Pamy.y_desired.shape[1], 'Something is wrong with the dimension of par_paiff_par_wff.'
        assert grads.shape[1] == idx_list[-1], 'Something is wrong with the dimension of par_paiff_par_wff.'
        par_paiff_par_wff.append(grads)

    # par_paiff_par_wff = get_par_paiff_par_wff(datapoints=datapoint, block_list=block_list, num_paras=idx_list[-1])
    # par_paid_par_wd = get_par_paiff_par_wff(datapoints=disturb_datapoint, block_list=disturb_block_list, num_paras=disturb_idx_list[-1])

    # Nt*nfb
    # par_paifb_par_wfb, par_paifb_par_fbin = get_par_paifb_par_wfb_and_ydiff(block_list=fb_block_list, fb_datapoints_list=fb_datasets, num_paras=fb_idx_list[-1])

    ### learnable PD update
    # def get_par_paifb_par_y(flag_dof, dim, par_paifb_par_fbin, dim_fb):
    #     par_paifb_par_y = []
    #     choosing_matrix = np.concatenate((np.zeros((dim_fb,dim)), np.eye(dim_fb), np.zeros((dim_fb,dim+1))), axis=1)
    #     for dof,flag in enumerate(flag_dof):
    #         if par_paifb_par_fbin[dof] is None:
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

    ### fixed PD update for LQR from MIMO SI
    # def get_par_paifb_par_y(flag_dof, dim, lqr):
    #     par_paifb_par_y = []
    #     for dof,flag in enumerate(flag_dof):
    #         if not flag:
    #             par_paifb_par_y.append(None)
    #         else:
    #             temp = np.zeros((dim,dim))
    #             k_pre_2 = lqr[dof, dof]
    #             k_pre_1 = lqr[dof, dof+3]
    #             for row in range(1,dim):
    #                 temp[row, row-1] = k_pre_1
    #                 if row>1:
    #                     temp[row, row-2] = k_pre_2
    #             par_paifb_par_y.append(temp)
    #     return par_paifb_par_y
    # # Nt x Nt            
    # par_paifb_par_y = get_par_paifb_par_y(flag_dof=paras.flag_dof, dim=Pamy.y_desired.shape[1], lqr=Pamy.lqr_k)

    # 1 x 3*Nt
    # TODO: need chenges for MIMO and for adding weights
    par_l_par_y = [((y_out-theta)[i]).reshape(1,-1) for i in range(3)]
    # idx_amp = [1,1.5,2]
    # for idx_ly in range(3):
    #     par_l_par_y[idx_ly][0,int(T_go_list[-1]*100-30):int(T_go_list[-1]*100+30)] *= idx_amp[idx_ly]
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

    # par_Bdd_par_wd = []
    # for dof,flag in enumerate(paras.flag_dof):
    #     if not flag:
    #         par_Bdd_par_wd.append(None)
    #     else:
    #         # temp = Bd_list[dof]@par_paid_par_wd[dof]
    #         temp = par_G_par_u[dof]@par_paid_par_wd[dof]
    #         par_Bdd_par_wd.append(temp)

    # Nt*nfb
    # par_y_par_wfb = []
    # for dof,flag in enumerate(paras.flag_dof):
    #     if not flag:
    #         par_y_par_wfb.append(None)
    #     else:
    #         temp = np.linalg.pinv(np.eye(Pamy.y_desired.shape[1])+par_G_par_u[dof]@par_paifb_par_y[dof])@par_G_par_u[dof]@par_paifb_par_wfb[dof]
    #         par_y_par_wfb.append(temp)

    def get_new_parameters(W_list, method, lr, par_l_par_y, par_y_par_w, par_pai_par_w, disturb=False):
        for dof in range(len(par_l_par_y)):
            if par_y_par_w[dof] is None:
                W_list[dof] = None
                continue
            if method == 'GD':
                # TODO: first ele in list of lr works for MIMO
                delta = (lr[dof]*par_l_par_y[dof]@par_y_par_w[dof]).reshape(-1, 1)
            elif method == 'NM':
                if disturb:
                    disturb_hessian_temp[dof] += par_y_par_w[dof].T@par_y_par_w[dof]+paras.disturb_alpha_list[dof]*par_pai_par_w[dof].T@par_pai_par_w[dof]+paras.disturb_epsilon_list[dof]*np.eye(par_pai_par_w[dof].shape[1])
                    # hessian_temp[dof] = par_y_par_w[dof].T@par_y_par_w[dof]+paras.alpha_list[dof]*par_pai_par_w[dof].T@par_pai_par_w[dof]+0.0*np.eye(par_pai_par_w[dof].shape[1])
                    delta = (lr[dof]*np.linalg.pinv(disturb_hessian_temp[dof]/(i_iter+1))@par_y_par_w[dof].T@par_l_par_y[dof].T).reshape(-1 ,1)
                else:
                    hessian_temp[dof] += (par_y_par_w[dof].T@par_y_par_w[dof]+paras.alpha_list[dof]*par_pai_par_w[dof].T@par_pai_par_w[dof]+paras.epsilon_list[dof]*np.eye(par_pai_par_w[dof].shape[1]))
                    # hessian_temp[dof].append(par_y_par_w[dof].T@par_y_par_w[dof]+paras.alpha_list[dof]*par_pai_par_w[dof].T@par_pai_par_w[dof]+paras.epsilon_list[dof]*np.eye(par_pai_par_w[dof].shape[1]))
                    # hessian_temp[dof].pop(0)
                    # hessian_temp[dof] = par_y_par_w[dof].T@par_y_par_w[dof]+paras.alpha_list[dof]*par_pai_par_w[dof].T@par_pai_par_w[dof]+0.0*np.eye(par_pai_par_w[dof].shape[1])
                    # if i_iter<10:
                    #     delta = (lr[dof]*scipy.linalg.pinv(sum(hessian_temp[dof])/(i_iter+1))@par_y_par_w[dof].T@par_l_par_y[dof].T).reshape(-1 ,1)
                    # else:
                    #     delta = (lr[dof]*scipy.linalg.pinv(sum(hessian_temp[dof])/10)@par_y_par_w[dof].T@par_l_par_y[dof].T).reshape(-1 ,1)
                    delta = (lr[dof]*np.linalg.pinv(hessian_temp[dof]/(i_iter+1))@par_y_par_w[dof].T@par_l_par_y[dof].T).reshape(-1 ,1)
            W_list[dof] = W_list[dof] - delta
        return W_list, delta  

    W_list, delta = get_new_parameters(W_list=W_list, method=paras.method_updating_policy, lr=paras.lr_list, par_l_par_y=par_l_par_y, par_y_par_w=par_y_par_wff, par_pai_par_w=par_paiff_par_wff)   
    block_list = set_parameters(block_list, W_list, idx_list, shape_list, device)
    # disturb_W_list, disturb_delta = get_new_parameters(W_list=disturb_W_list, method=paras.method_updating_policy, lr=[20, 50, 50], par_l_par_y=par_l_par_y, par_y_par_w=par_Bdd_par_wd, par_pai_par_w=par_paid_par_wd, disturb=True)   
    # disturb_block_list = set_parameters(disturb_block_list, disturb_W_list, disturb_idx_list, disturb_shape_list, device)
    # TODO: an unsolved bug after adding learnbale fb
    # TODO: an unsolved bug, also need to adapt for ff blocks
    # if i_iter==0:
    #     W_list, delta = get_new_parameters(W_list=W_list, method=paras.method_updating_policy, lr=[0, 0, 0], par_l_par_y=par_l_par_y, par_y_par_w=par_y_par_wff, par_pai_par_w=par_paiff_par_wff)   
    #     block_list = set_parameters(block_list, W_list, idx_list, shape_list, device)
    #     fb_W_list, fb_delta = get_new_parameters(W_list=fb_W_list, method='GD', lr=[0, 0, 0], par_l_par_y=par_l_par_y, par_y_par_w=par_y_par_wfb, par_pai_par_w=par_paifb_par_wfb)   
    #     fb_block_list = set_parameters(fb_block_list, fb_W_list, fb_idx_list, fb_shape_list, device)
    # else:
    #     W_list, delta = get_new_parameters(W_list=W_list, method=paras.method_updating_policy, lr=paras.lr_list, par_l_par_y=par_l_par_y, par_y_par_w=par_y_par_wff, par_pai_par_w=par_paiff_par_wff)   
    #     block_list = set_parameters(block_list, W_list, idx_list, shape_list, device)
    #     fb_W_list, fb_delta = get_new_parameters(W_list=fb_W_list, method='GD', lr=[10000, 50000, 50000], par_l_par_y=par_l_par_y, par_y_par_w=par_y_par_wfb, par_pai_par_w=par_paifb_par_wfb)   
    #     fb_block_list = set_parameters(fb_block_list, fb_W_list, fb_idx_list, fb_shape_list, device)

    if (i_iter+1)%50==0:
        root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(paras.save_path_num) +'/linear_model_ff' + '/' + str(i_iter+1)
        mkdir(root_model_epoch) 
        for dof,cnn in enumerate(block_list):
            if cnn is None:
                continue
            root_file = root_model_epoch + '/' + str(dof)
            torch.save(cnn.state_dict(), root_file)
        # disturb_root_model_epoch = '/home/mtian/Desktop/MPI-intern/training_log_temp_'+ str(paras.save_path_num) +'/linear_model_disturb' + '/' + str(i_iter+1)
        # mkdir(disturb_root_model_epoch) 
        # for dof,cnn in enumerate(disturb_block_list):
        #     if cnn is None:
        #         continue
        #     disturb_root_file = disturb_root_model_epoch + '/' + str(dof)
        #     torch.save(cnn.state_dict(), disturb_root_file)
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

    part1_temp = (y_out-theta)
    loss = [np.linalg.norm(part1_temp[i].reshape(1,-1)) for i in range(len(part1_temp))]
    print('loss:')
    print(loss)
    if paras.flag_wandb:
        run.log({'log(loss_0)': np.log10(loss[0]/t_stamp[-1]/math.pi*180/100), 'log(loss_1)': np.log10(loss[1]/t_stamp[-1]/math.pi*180/100), 'log(loss_2)': np.log10(loss[2]/t_stamp[-1]/math.pi*180/100)}, i_iter+1)
        run.log({'visualization': plots_to_show})

    i_iter += 1

    if paras.flag_time_analysis:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
