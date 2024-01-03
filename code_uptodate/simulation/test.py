'''
This script is used to train the robot with online convex optimization
single trajectory
'''
import PAMY_CONFIG
import math
import os
import numpy as np
import o80
import o80_pam
import matplotlib.pyplot as plt
import pickle5 as pickle
# %%
# Geometry                   = PAMY_CONFIG.build_geometry()                      
# (time_list, position_list) = PAMY_CONFIG.get_recorded_ball(Geometry=Geometry)  # get the list of interception time and position
# T_back           = 1.5
# T_steady         = 0.2
# train_index      = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
# test_index       = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
# weight_list      = [11, 13, 14, 48, 56, 58, 59, 60, 61, 65, 68]

# def mkdir(path):
#     folder = os.path.exists(path)
#     if not folder:
#         os.makedirs(path)

# def get_weight(i):
#     if i in weight_list:
#         weight = (10.0, 8.0)
#     else:
#         weight = (6.0, 10.0)
#     return weight
# def get_trajectory(T=None, target=None, T_back=1.5, T_steady=0.2, if_plot='no', plan_weight=(6, 10)):
#     (p_mjp, p_angular_mjp, t_stamp) = Geometry.PathPlanning(time_point=0, T_go=T, T_back=T_back, angle=np.array([0.000000, -0.514884, -0.513349, 0.0]),
#                                                             T_steady=T_steady, velocity_initial=np.array([0, 0, 0]), 
#                                                             acceleration_initial=np.array([0, 0, 0]),
#                                                             target=target, plan_weight=plan_weight)
        
#     p_temp = p_angular_mjp
#     p_temp = p_temp - p_temp[:, 0].reshape(-1, 1)
#     p_last_dof = np.zeros(len(t_stamp)).reshape(1, -1)
#     p = np.vstack((p_temp, p_last_dof))
#     p[1, :] = -p[1, :]
#     p[2, :] = -p[2, :]
    
#     # fig = plt.figure(figsize=(16, 16))
#     # for i in range(4):
#     #     plt.plot(t_stamp, p[i, :]*180/math.pi, label=r'dof {}'.format(i))
#     # plt.legend(ncol=3)
#     # plt.show()

#     return (p, t_stamp)

# path = "/home/hao/Desktop/Learning/data/online_optimization/train"
# mkdir(path)

# for index in train_index:
#     plan_weight = get_weight(index)
#     T           = time_list[index]
#     target = Geometry.EndToAngle(position_list[index], frame='Cartesian')    
#     (p, t) = get_trajectory(T, target, plan_weight=plan_weight)
    
#     path_file = path + '/' + str(index) 
#     file = open(path_file, 'wb')
#     pickle.dump(t, file, -1)
#     pickle.dump(p, file, -1)
#     file.close()

# path = "/home/hao/Desktop/Learning/data/online_optimization/test"
# mkdir(path)

# for index in test_index:
#     plan_weight = get_weight(index)
#     T           = time_list[index]
#     target = Geometry.EndToAngle(position_list[index], frame='Cartesian')    
#     (p, t) = get_trajectory(T, target, plan_weight=plan_weight)
    
#     path_file = path + '/' + str(index) 
#     file = open(path_file, 'wb')
#     pickle.dump(t, file, -1)
#     pickle.dump(p, file, -1)
#     file.close()

