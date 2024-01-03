'''
This script is used compare the tracking performance of different methods
'''
import torch
import math
import os
import numpy as np
import o80
import o80_pam
import PAMY_CONFIG
import pickle5 as pickle
import CNNs
import matplotlib.pyplot as plt
# %%
index_list = [11, 13, 14, 48, 56, 58, 59, 60, 61, 65, 68]
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def three_courses(y_des, ff, u_ago, u_ant):
    y_des = y_des - y_des[:, 0].reshape(-1, 1)
    y   = [None] * 3
    fb  = [None] * 3
    ago = [None] * 3
    ant = [None] * 3
    mode_name_list = ['ff', 'ff', 'ff', 'ff']
    [y[0], fb[0], ago[0], ant[0]] = Pamy.Control(y_list=y_des, mode_name_list=mode_name_list, 
                                                 mode_trajectory="ref", ifplot="no", 
                                                 u_ago=u_ago, u_ant=u_ant, ff=ff, echo="yes")
    
    Pamy.AngleInitialization(Geometry.initial_posture)
    # Pamy.PressureInitialization()
    
    mode_name_list = ['ff+fb', 'ff+fb', 'ff+fb', 'ff+fb']
    [y[1], fb[1], ago[1], ant[1]] = Pamy.Control(y_list=y_des, mode_name_list=mode_name_list, 
                                                 mode_trajectory="ref", ifplot="no", 
                                                 u_ago=u_ago, u_ant=u_ant, ff=ff, echo="yes", 
                                                 controller='pd')
    Pamy.AngleInitialization(Geometry.initial_posture)
    # Pamy.PressureInitialization()

    # [y[2], fb[2], ago[2], ant[2]] = Pamy.Control(y_list=y_des, mode_name_list=mode_name_list, 
    #                                              mode_trajectory="ref", ifplot="no", 
    #                                              u_ago=u_ago, u_ant=u_ant, ff=ff, echo="yes", 
    #                                              controller='pid')
    # Pamy.AngleInitialization(Geometry.initial_posture)
    # Pamy.PressureInitialization()
    
    return (y, fb, ago, ant)
# %%
frontend                   = o80_pam.FrontEnd("real_robot")
Geometry                   = PAMY_CONFIG.build_geometry()
Pamy                       = PAMY_CONFIG.build_pamy(frontend=frontend)
(time_list, position_list) = PAMY_CONFIG.get_recorded_ball(Geometry=Geometry)
# %% initilization
Pamy.AngleInitialization(Geometry.initial_posture)
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
# %% read cnn model and linear model from file
model_3_ch = CNNs.get_model_3_ch()
(model_2_ch, linear_model) = CNNs.get_model_2_ch()
# %% online optimization for all dofs
verify_data = '/home/hao/Desktop/HitBall/data/all_methods'
mkdir(verify_data)

for index in range(1, 2):
    if np.linalg.norm( position_list[index, :], ord=2 ) - (Geometry.l_1 + Geometry.l_2) > 1e-10:
        print('{}. ball can not be reached'.format( index ))
    else:
        if (index != 20) and (index != 69):

            y_des_list = [None] * 5
            ff_list    = [None] * 5
            y_list     = [None] * 5
            fb_list    = [None] * 5
            ago_list   = [None] * 5
            ant_list   = [None] * 5

            if index in index_list:
                plan_weight = (10.0, 8.0)
            else:
                plan_weight = (6.0, 10.0)

            T = time_list[index]
            T_back = 1.5
            T_steady = 0.2
            target = Geometry.EndToAngle(position_list[index], frame='Cartesian')
            (p, p_angular, t_stamp) = Geometry.PathPlanning(time_point=0, T_go=T, T_back=T_back, 
                                                            T_steady=T_steady, angle=angle_initial_read, 
                                                            target=target, plan_weight=plan_weight )
            p_angular = p_angular - p_angular[:, 0].reshape(-1, 1)
            last_dof = np.zeros(len(t_stamp)).reshape(1, -1)
            p_angular = np.vstack((p_angular, last_dof))
            '''
            list:
            1. recorded ilc
            2. linear control framework
            3. linear regression model
            4. 2-channel cnn model
            5. 3-channel cnn model
            nake sure all the list is in 4*length
            '''
            # (y_des_list[0], ff_list[0]) = CNNs.get_ilc_ff(index)

            # ff_list[1] = np.vstack((CNNs.get_linear_ff(linear_model, np.copy(p_angular[0:3, :])), last_dof))

            # [u_ago, u_ant, ff_list[2]] = Pamy.Feedforward(y_list=p_angular)
            [u_ago, u_ant, _] = Pamy.Feedforward(y_list=p_angular)

            # ff_list[3] = np.vstack((CNNs.get_cnn_ff((model_2_ch, linear_model), np.copy(p_angular[0:3, :])), last_dof))
            
            ff_list[0] = np.vstack((CNNs.get_cnn_ff((model_3_ch, ), np.copy(p_angular[0:3, :])), last_dof))

            for i in range(0, 1):
                y_des_list[i] = p_angular + angle_initial_read.reshape(-1, 1)
            # check the fixed reference trajectory
            
            fig = plt.figure(figsize=(16, 16))
            for i in range(3):
                plt.plot(t_stamp, ff_list[0][i, :], label=r'dof {}'.format(i))
                # plt.plot(t_stamp, y_des_list[0][i, :]*180/math.pi, linestyle='--', linewidth=2.0, label=r'dof {}'.format(i))
            plt.legend(ncol=3)
            plt.show()
      

            # fig, axs = plt.subplots(5, 3, figsize=(40, 20))
            # for i_dof in range(3):
            #     line = []
            #     for i_row in range(5):
            #         ax = axs[i_row, i_dof]
            #         ax.set_xlabel(r'Time $t$ in s')
            #         ax.set_ylabel(r'Normalized Feedforward')
            #         line_temp, = ax.plot(t_stamp, ff_list[i_row][i_dof, :], linewidth=1.5, linestyle='-', label=r'index {}'.format(i_row))
            #         line.append(line_temp)
            #         ax.grid()
            # ax.legend(handles=line)
            # plt.show()

            # control the  
            for i in range(1):
                [y_list[i], fb_list[i], ago_list[i], ant_list[i]] = three_courses(np.copy(y_des_list[i]), ff_list[i], u_ago, u_ant)

            t_list = np.array([0, T, T+T_back, T+T_back+T_steady])
            path_of_file = verify_data + '/' + str(index)

            # file = open(path_of_file, 'wb')
            # pickle.dump(t_stamp, file, -1)
            # pickle.dump(t_list, file, -1)
            # pickle.dump(y_des_list, file, -1)
            # pickle.dump(y_list, file, -1)
            # pickle.dump(ff_list, file, -1)
            # pickle.dump(fb_list, file, -1)
            # pickle.dump(ago_list, file, -1)
            # pickle.dump(ant_list, file, -1)
            # file.close()

            angle_initial_read = np.array(frontend.latest().get_positions())