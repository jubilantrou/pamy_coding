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
import LimitCheck as LC
import random
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
# %%
frontend         = o80_pam.FrontEnd("real_robot")  # connect to the real robot
Pamy             = PAMY_CONFIG.build_pamy(frontend=frontend)
Geometry         = PAMY_CONFIG.build_geometry()   
root             = "/home/hao/Desktop/Learning/online_convex_optimization"
alpha_list        = [1e-9, 3e-10, 1e-9]
epsilon_list      = [1e-9, 1e-10, 1e-9]
step_size_list    = [0.5, 0.5, 0.5]
version           = 'v_1'  # v_1 is single method, v_2 consider the past information
projection        = 'osqp'
learning_mode     = 'b'
mode_name         = 'ff'
coupling          = 'no'
step_size_version = 'sqrt'
number_iteration  = 200  # train iterations for each trajectory
h                 = 100
T_back            = 1.5
T_steady          = 0.2
# %%
train_index       = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
test_index        = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
# %%
Pamy.AngleInitialization(Geometry.initial_posture)
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
# %% build Pamy for training and testing
t_train_list = []
t_test_list  = []
Pamy_train   = []
Pamy_test    = []

for index in train_index:
    Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
    (t, p) = PAMY_CONFIG.get_trajectory(index, mode='train')
    T      = t[-1]
    t_train_list.append(np.array([0, T, T+1.5, T+1.7]))
    Pamy.ImportTrajectory(p, t)  #  import the desired trajectories and the time stamp
    Pamy.GetOptimizer_convex(angle_initial_read, h=h, mode_name='none', coupling=coupling, learning_mode=learning_mode)
    Pamy_train.append(Pamy)

for index in test_index:
    Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
    (t, p) = PAMY_CONFIG.get_trajectory(index, mode='test')
    T      = t[-1]
    t_test_list.append(np.array([0, T, T+1.5, T+1.7]))
    Pamy.ImportTrajectory(p, t)
    Pamy.GetOptimizer_convex(angle_initial_read, h=h, mode_name='none', coupling=coupling, learning_mode=learning_mode)
    Pamy_test.append(Pamy)

# %% Model
root_model = root + '/' + 'model'  
root_file  = root_model + '/' + 'model' 
file = open(root_file, 'rb')
b_list = pickle.load(file)
file.close()
# %% Train
root_train = root + '/' + 'train'
mkdir(root_train)
root_train_ff = root_train + '/' + 'ff'
mkdir(root_train_ff)
root_train_fb = root_train + '/' + 'fb'
mkdir(root_train_fb)

for i in range(len(Pamy_train)):

    Pamy    = Pamy_train[i]
    t       = t_train_list[i]
    t_stamp = Pamy.t_stamp
    
    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, 
                                                                    mode_name='ff', 
                                                                    coupling=coupling, 
                                                                    learning_mode=learning_mode)
    
    root_file = root_train_ff + '/' + str(train_index[i]) 
    file = open(root_file, 'wb')
    pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    pickle.dump(t, file, -1)
    pickle.dump(angle_initial_read, file, -1)
    pickle.dump(y, file, -1)
    pickle.dump(Pamy.y_desired, file, -1)
    pickle.dump(ff, file, -1)
    pickle.dump(fb, file, -1)
    pickle.dump(obs_ago, file, -1)
    pickle.dump(obs_ant, file, -1)
    pickle.dump(b_list, file, -1)
    file.close()

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    Pamy.PressureInitialization()
    angle_initial_read =np.array(frontend.latest().get_positions())

    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, 
                                                                    mode_name='ff+fb', 
                                                                    coupling=coupling, 
                                                                    learning_mode=learning_mode)
    
    root_file = root_train_fb + '/' + str(train_index[i]) 
    file = open(root_file, 'wb')
    pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    pickle.dump(t, file, -1)
    pickle.dump(angle_initial_read, file, -1)
    pickle.dump(y, file, -1)
    pickle.dump(Pamy.y_desired, file, -1)
    pickle.dump(ff, file, -1)
    pickle.dump(fb, file, -1)
    pickle.dump(obs_ago, file, -1)
    pickle.dump(obs_ant, file, -1)
    pickle.dump(b_list, file, -1)
    file.close()

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    Pamy.PressureInitialization()
    angle_initial_read =np.array(frontend.latest().get_positions())

# %% Test
root_test = root + '/' + 'test'
mkdir(root_test)
root_test_ff = root_test + '/' + 'ff'
mkdir(root_test_ff)
root_test_fb = root_test + '/' + 'fb'
mkdir(root_test_fb)

for i in range(len(Pamy_test)):

    Pamy    = Pamy_test[i]
    t       = t_test_list[i]
    t_stamp = Pamy.t_stamp
    
    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, 
                                                                    mode_name='ff', 
                                                                    coupling=coupling, 
                                                                    learning_mode=learning_mode)
    
    root_file = root_test_ff + '/' + str(test_index[i]) 
    file = open(root_file, 'wb')
    pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    pickle.dump(t, file, -1)
    pickle.dump(angle_initial_read, file, -1)
    pickle.dump(y, file, -1)
    pickle.dump(Pamy.y_desired, file, -1)
    pickle.dump(ff, file, -1)
    pickle.dump(fb, file, -1)
    pickle.dump(obs_ago, file, -1)
    pickle.dump(obs_ant, file, -1)
    pickle.dump(b_list, file, -1)
    file.close()

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    Pamy.PressureInitialization()
    angle_initial_read =np.array(frontend.latest().get_positions())

    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, 
                                                                    mode_name='ff+fb', 
                                                                    coupling=coupling, 
                                                                    learning_mode=learning_mode)
    
    root_file = root_test_fb + '/' + str(test_index[i]) 
    file = open(root_file, 'wb')
    pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    pickle.dump(t, file, -1)
    pickle.dump(angle_initial_read, file, -1)
    pickle.dump(y, file, -1)
    pickle.dump(Pamy.y_desired, file, -1)
    pickle.dump(ff, file, -1)
    pickle.dump(fb, file, -1)
    pickle.dump(obs_ago, file, -1)
    pickle.dump(obs_ant, file, -1)
    pickle.dump(b_list, file, -1)
    file.close()

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    Pamy.PressureInitialization()
    angle_initial_read =np.array(frontend.latest().get_positions())