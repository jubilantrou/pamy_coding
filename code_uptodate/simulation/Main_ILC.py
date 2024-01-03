'''
This script is used to train the robot with ILC
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
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
# %%
train_index      = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
test_index       = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
index_list       = [17]
frontend         = o80_pam.FrontEnd("real_robot")
Pamy             = PAMY_CONFIG.build_pamy(frontend=frontend)
(t_stamp, p)     = PAMY_CONFIG.get_trajectory(index_list[0])
number_iteration = 30
root             = '/home/hao/Desktop/Learning'
root_data        = root + '/' + 'data' + '/' + 'comparison'
mkdir(root_data)
# %%
Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
# %% train the ilc
for index in index_list:
    T = t_stamp[-1] - 1.7
    T_back = 1.5
    T_steady = 0.2
    '''
    1. update the desired trajectories
    2. update the optimizer for each dof
    3. train the robot with ILC algorithm
    '''
    Pamy.ImportTrajectory(p, t_stamp)
    Pamy.GetOptimizer(angle_initial_read, total_iteration=number_iteration, mode_name='none')
    (y_history, repeated, ff_history, disturbance_history, \
    P_history, d_lifted_history, P_lifted_history, \
    fb_history, ago_history, ant_history, y_pid) = Pamy.ILC(number_iteration=number_iteration, 
                                                            GLOBAL_INITIAL=PAMY_CONFIG.GLOBAL_INITIAL,
                                                            mode_name='none')

    t_list = np.array([0, T, T+T_back, T+T_back+T_steady])

    root_file = root_data + '/' + str(index) + '_ILC'
    file = open(root_file, 'wb')
    pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    pickle.dump(t_list, file, -1)
    pickle.dump(angle_initial_read, file, -1)
    pickle.dump(y_history, file, -1)
    pickle.dump(repeated, file, -1)
    pickle.dump(y_pid, file, -1)
    pickle.dump(ff_history, file, -1)
    pickle.dump(fb_history, file, -1)
    pickle.dump(ago_history, file, -1)
    pickle.dump(ant_history, file, -1)
    pickle.dump(disturbance_history, file, -1)
    pickle.dump(P_history, file, -1)
    pickle.dump(d_lifted_history, file, -1)
    pickle.dump(P_lifted_history, file, -1)
    file.close()

    angle_initial_read =np.array(frontend.latest().get_positions())