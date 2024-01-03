'''
This script is used to train the robot with online convex optimization
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
frontend                   = o80_pam.FrontEnd("real_robot")                    # connect to the real robot
Pamy                       = PAMY_CONFIG.build_pamy(frontend=frontend)
Geometry                   = PAMY_CONFIG.build_geometry()                      
(time_list, position_list) = PAMY_CONFIG.get_recorded_ball(Geometry=Geometry)  # get the list of interception time and position
# %%
number_iteration_1 = 15  # train iterations for each trajectory
number_iteration_2 = 15
h_1                = 1
h_2                = 3
T_back             = 1.5
T_steady           = 0.2
# train_index      = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
train_index        = [17]
test_index         = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
weight_list        = [11, 13, 14, 48, 56, 58, 59, 60, 61, 65, 68]
Pamy.AngleInitialization(Geometry.initial_posture)
# Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_weight(i):
    if i in weight_list:
        weight = (10.0, 8.0)
    else:
        weight = (6.0, 10.0)
    return weight

def get_trajectory(T=None, target=None, T_back=1.5, T_steady=0.2, if_plot='no'):
    (p_mjp, p_angular_mjp, t_stamp) = Geometry.PathPlanning(time_point=0, T_go=T, T_back=T_back, 
                                                            T_steady=T_steady, angle=angle_initial_read, 
                                                            velocity_initial=np.array([0, 0, 0]), 
                                                            acceleration_initial=np.array([0, 0, 0]),
                                                            target=target)
        
    p_temp = p_angular_mjp
    p_temp = p_temp - p_temp[:, 0].reshape(-1, 1)
    
    if if_plot=='yes':
        fig = plt.figure(figsize=(16, 16))
        for i in range(3):
            plt.plot(t_stamp, p_mjp[i, :], label=r'dof {}'.format(i))
        plt.legend(ncol=3)
        plt.show()

        fig = plt.figure(figsize=(16, 16))
        for i in range(3):
            plt.plot(t_stamp, p_angular_mjp[i, :]*180/math.pi, label=r'dof {}'.format(i))
        plt.legend(ncol=3)
        plt.show()

        fig = plt.figure(figsize=(16, 16))
        for i in range(3):
            plt.plot(t_stamp, p_temp[i, :]*180/math.pi, label=r'dof {}'.format(i))
        plt.legend(ncol=3)
        plt.show()

    # do not control last dof
    p_last_dof = np.zeros(len(t_stamp)).reshape(1, -1)
    p_temp = np.vstack((p_temp, p_last_dof))

    return (p_temp, t_stamp)

def get_initial_guess(Pamys=None):
    b_list = []
    for dof in range(3):
        part1 = np.copy(Pamys[0].part1_list[dof])
        part2 = np.copy(Pamys[0].part2_list[dof])

        for i in range(1, len(Pamys)):
            part1 += Pamys[i].part1_list[dof]
            part2 += Pamys[i].part2_list[dof]
        
        b = np.linalg.pinv(part1)@part2
        b_list.append(b)
    return b_list
# %% online convex optimization
root = "/home/hao/Desktop/Learning/online_convex_optimization"
mkdir(root)
Pamy_train = []
Pamy_train_extend = []
t_list = []

for index in train_index:
    
    Pamy        = PAMY_CONFIG.build_pamy(frontend=frontend)
    Pamy_extend = PAMY_CONFIG.build_pamy(frontend=frontend)
    plan_weight = get_weight(index)
    T = time_list[index]
    t_list.append(np.array([0, T, T+1.5, T+1.7]))
    '''
    This function should have been updated, but was not updated for consistency with before.
    '''
    target = Geometry.EndToAngle(position_list[index], frame='Cartesian')    
    (p, t) = get_trajectory(T, target)
    '''
    import the desired trajectories and the time stamp
    '''
    Pamy.ImportTrajectory(p, t)
    Pamy_extend.ImportTrajectory(p, t)
    Pamy.GetOptimizer_convex(angle_initial_read, h=h_1, mode_name='pd')
    Pamy_extend.GetOptimizer_convex(angle_initial_read, h=h_2, mode_name='pd')
    Pamy_train.append(Pamy)
    Pamy_train_extend.append(Pamy_extend)
# %% Train
b_list = get_initial_guess(Pamy_train)
root_train = root + '/' + 'train'
n0 = 5
n1 = 10
k  = 1
mkdir(root_train)
mode_name = 'ff+fb'

for it in range(number_iteration_1):
    
    for iPamy in range(len(Pamy_train)):

        root_train_round = root_train + '/' + str(train_index[iPamy])
        mkdir(root_train_round)
        
        Pamy    = Pamy_train[iPamy]
        t       = t_list[iPamy]
        t_stamp = Pamy.t_stamp
        (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, mode_name=mode_name)
        
        root_file = root_train_round + '/' + str(it) 
        file = open(root_file, 'wb')
        pickle.dump(t_stamp, file, -1) # time stamp for x-axis
        pickle.dump(t_list, file, -1)
        pickle.dump(angle_initial_read, file, -1)
        pickle.dump(y, file, -1)
        pickle.dump(Pamy.y_desired, file, -1)
        pickle.dump(ff, file, -1)
        pickle.dump(fb, file, -1)
        pickle.dump(obs_ago, file, -1)
        pickle.dump(obs_ant, file, -1)
        pickle.dump(b_list, file, -1)
        file.close()

        step_size = n0/(n0+n1 * np.sqrt(k/5))
        k += 1

        y_out = y - y[:, 0].reshape(-1, 1)
        for i in range(len(b_list)):
            b_list[i] = b_list[i] - step_size * Pamy.gradient_list[i]@(y_out[i, :].reshape(-1, 1) - Pamy.y_desired[i, :].reshape(-1, 1))

        Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
        angle_initial_read =np.array(frontend.latest().get_positions())

for i in range(len(b_list)):
    b_list[i] = np.linalg.pinv(Pamy_extend.O_lsit[i].Xi)@Pamy.O_lsit[i].Xi@b_list[i]

for it in range(number_iteration_1, number_iteration_1+number_iteration_2):
    
    for iPamy in range(len(Pamy_train_extend)):

        root_train_round = root_train + '/' + str(train_index[iPamy])
        mkdir(root_train_round)
        
        Pamy    = Pamy_train_extend[iPamy]
        t       = t_list[iPamy]
        t_stamp = Pamy.t_stamp
        (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, mode_name=mode_name)
        
        root_file = root_train_round + '/' + str(it) 
        file = open(root_file, 'wb')
        pickle.dump(t_stamp, file, -1) # time stamp for x-axis
        pickle.dump(t_list, file, -1)
        pickle.dump(angle_initial_read, file, -1)
        pickle.dump(y, file, -1)
        pickle.dump(Pamy.y_desired, file, -1)
        pickle.dump(ff, file, -1)
        pickle.dump(fb, file, -1)
        pickle.dump(obs_ago, file, -1)
        pickle.dump(obs_ant, file, -1)
        pickle.dump(b_list, file, -1)
        file.close()

        step_size = n0/(n0+n1 * np.sqrt(k))
        k += 1

        y_out = y - y[:, 0].reshape(-1, 1)
        for i in range(len(b_list)):
            b_list[i] = b_list[i] - step_size * Pamy.gradient_list[i]@(y_out[i, :].reshape(-1, 1) - Pamy.y_desired[i, :].reshape(-1, 1))

        Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
        angle_initial_read =np.array(frontend.latest().get_positions())
# %% Test
# Pamy_test = []
# root_test = root + '/' + 'test'
# mkdir(root_test)
# for index in test_index:
#     Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
#     plan_weight = get_weight(index)
#     T = time_list[index]
#     '''
#     This function should have been updated, but was not updated for consistency with before.
#     '''
#     target = Geometry.EndToAngle(position_list[index], frame='Cartesian')    
#     (p, t) = get_trajectory(T, target)
#     Pamy.ImportTrajectory(p, t)
#     Pamy.GetOptimizer_convex(angle_initial_read, mode_name='pd')
#     (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, mode_name=mode_name)

#     root_file = root_test + '/' + str(index) 
#     file = open(root_file, 'wb')
#     pickle.dump(t_stamp, file, -1) # time stamp for x-axis
#     pickle.dump(t_list, file, -1)
#     pickle.dump(angle_initial_read, file, -1)
#     pickle.dump(y, file, -1)
#     pickle.dump(Pamy.y_desired, file, -1)
#     pickle.dump(ff, file, -1)
#     pickle.dump(fb, file, -1)
#     pickle.dump(obs_ago, file, -1)
#     pickle.dump(obs_ant, file, -1)
#     pickle.dump(b_list, file, -1)
#     file.close()

#     Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
#     angle_initial_read =np.array(frontend.latest().get_positions())

