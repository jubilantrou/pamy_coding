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
from scipy.optimize import minimize
import random
import osqp
from scipy import sparse
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_initial_guess(Pamys=None):
    b_list = []
    for dof in range(3):
        part1 = np.copy(Pamys[0].hessian_list[dof])
        part2 = np.copy(Pamys[0].part2_list[dof])
        b = np.linalg.pinv(part1)@part2
        b_list.append(b)
    return b_list

def get_osqp(dof, Xi, b):
    # limit_max = np.array([4500, 6500, 6000, 7000])
    # limit_min = np.array([-4500, -5000, -6000, -7000])
    """
    max_pressure_ago = [22000, 25000, 22000, 22000]
    max_pressure_ant = [22000, 23000, 22000, 22000]

    min_pressure_ago = [13000, 13500, 10000, 8000]
    min_pressure_ant = [13000, 14500, 10000, 8000]
    """
   
    limit_max = np.array([ 4500,  4300,  6000,  7000])
    limit_min = np.array([-4500, -7200, -6000, -7000])
    
    m = osqp.OSQP()
    (r, nr) = Xi.shape

    l = np.ones(r) * limit_min[dof] 
    u = np.ones(r) * limit_max[dof]

    q = -(Xi.T@Xi@b).flatten()
    A = sparse.csc_matrix(Xi)
    P = sparse.csc_matrix(Xi.T@Xi)

    m.setup(P=P, q=q, A=A, l=l, u=u)
    results = m.solve()
    return results.x.reshape(-1, 1)

def get_hessian(sum_1, sum_2, alpha, epsilon):
    (l, r) = sum_1.shape
    I = np.eye(l)
    hessian = sum_1 + alpha * sum_2 + epsilon * I
    return hessian

def get_update(dof, b, sk, hessian, gradient, Xi):
    print('dof{}: max b {}, max update {}'.format(dof, np.max(abs(b)), np.max(abs(sk * np.linalg.pinv(hessian) @ gradient))))
    # print('dof{}: max u update {}'.format(dof, np.max(abs(sk * Xi@np.linalg.pinv(hessian)@gradient))))
    b = b - sk * np.linalg.pinv(hessian) @ gradient
    return b

def get_projection(dof, b, Xi, coupling='no', projection='osqp'):
    if projection == 'cutoff':
        if learning_mode == 'b':
            b = np.linalg.pinv(Xi) @ LC.LimitCheck(Xi@b, dof)
        elif learning_mode == 'u':
            b = LC.LimitCheck(b, dof)
    elif projection == 'osqp':
        b = get_osqp(dof, Xi, b)
    return b

def get_newton_method(nr, dof, robot, sum_1_list, sum_2_list, y, alpha_list, epsilon_list, learning_mode='b'):
    alpha    = alpha_list[dof]
    epsilon  = epsilon_list[dof]
    sum_1    = sum_1_list[dof]
    sum_2    = sum_2_list[dof]
    L        = robot.gradient_list[dof]
    sum_1   += L.T@L
    Xi       = robot.Xi_list[dof]
    sum_2   += Xi.T@Xi
    if learning_mode == 'u':
        sum_2 = 0
    hessian  = get_hessian(sum_1/nr, sum_2/nr, alpha=alpha, epsilon=epsilon)
    gradient = L.T@(y[dof, :].reshape(-1, 1) - robot.y_desired[dof, :].reshape(-1, 1)) 
    return (hessian, gradient, sum_1, sum_2, Xi)

def get_step_size(nr, dof, step_size_version='constant'):
    factor = [2.0, 2.0, 2.0]
    step_size_list = [2, 2, 2]
    if step_size_version == 'constant':
        step_size = step_size_list[dof]
    elif step_size_version == 'sqrt':
        step_size = factor[dof]/(2+np.sqrt(nr))
        if dof == 1:
            step_size = 2.0
    return step_size 
# %%
frontend         = o80_pam.FrontEnd("real_robot")  # connect to the real robot
Pamy             = PAMY_CONFIG.build_pamy(frontend=frontend)
root             = '/home/hao/Desktop/Learning'
mkdir(root) 
# %%
alpha_list        = [1e-8,  1e-8, 1e-9] # u, ff
epsilon_list      = [1e-13, 0.0, 0.0]
# alpha_list        = [1e-8,  1e-8, 1e-8]  # useless in u learning mode
# epsilon_list      = [1e-9, 1e-9, 1e-9]
projection        = 'osqp'
learning_mode     = 'b'
mode_name         = 'ff'
coupling          = 'yes'
step_size_version = 'sqrt'
number_iteration  = 20 # train iterations for each trajectory
h                 = 100
T_back            = 1.5
T_steady          = 0.2
nr_channel        = 1
train_index       = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
test_index        = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
folder_name       = 'oco_single' + '_' + learning_mode + '_' + mode_name  + '_' + step_size_version + '_' + projection + '_' + coupling + '_' + str(nr_channel) + 'channel'
root_folder       = root + '/' + 'data' + '/' + folder_name
mkdir(root_folder)
index             = 17
root_learning     = root_folder + '/' + str(index) + '/' + 'learning'
mkdir(root_learning)
root_model        = root_folder + '/' + str(index) + '/' + 'model'
mkdir(root_model)   
# %%
Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
# %% build Pamy for training and testing
if index in train_index:
    mode = 'train'
elif index in test_index:
    mode = 'test'
Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
(t, p) = PAMY_CONFIG.get_trajectory(index, mode=mode)
T      = t[-1]
t_list = np.array([0, T, T+1.5, T+1.7])
Pamy.ImportTrajectory(p, t)  #  import the desired trajectories and the time stamp
Pamy.GetOptimizer_convex(angle_initial_read, h=h, coupling=coupling)
if learning_mode == 'u':
    sum_1_list = [np.zeros((p.shape[1], p.shape[1]))] * 3
elif learning_mode == 'b':
    if coupling == 'no':
        sum_1_list = [np.zeros((nr_channel*(h+h+1)+1, nr_channel*(h+h+1)+1))] * 3
        sum_2_list = [np.zeros((nr_channel*(h+h+1)+1, nr_channel*(h+h+1)+1))] * 3
    elif coupling == 'yes':
        sum_1_list = [np.zeros((3*(h+h+1)+1, 3*(h+h+1)+1))] * 3
        sum_2_list = [np.zeros((3*(h+h+1)+1, 3*(h+h+1)+1))] * 3
# %% Train
for it in range(number_iteration):
    if it == 0:
        b_list = get_initial_guess([Pamy])
    
    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, coupling=coupling)
    y_out = y - y[:, 0].reshape(-1, 1)

    root_file = root_learning + '/' + str(it) 
    file = open(root_file, 'wb')
    pickle.dump(Pamy.t_stamp, file, -1)
    pickle.dump(t_list, file, -1)
    pickle.dump(angle_initial_read, file, -1)
    pickle.dump(y, file, -1)
    pickle.dump(Pamy.y_desired, file, -1)
    pickle.dump(ff, file, -1)
    pickle.dump(fb, file, -1)
    pickle.dump(obs_ago, file, -1)
    pickle.dump(obs_ant, file, -1)
    pickle.dump(b_list, file, -1)
    pickle.dump(index, file, -1)
    file.close()

    for dof in range(len(b_list)):  # update the linear model b
        '''
        b = b - s_k * pinv(1/t*sum(L.T * L)+alpha/t*sum(X.T * X)+epsilon*I) * L.T * (y_out - y_des)
        '''
        [hessian, gradient, sum_1_list[dof], sum_2_list[dof], Xi] = get_newton_method(it+1, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list)
        sk                                                        = get_step_size(it+1, dof, step_size_version=step_size_version)
        b_list[dof]                                               = get_update(dof, b_list[dof], sk, hessian, gradient, Xi)
        b_list[dof]                                               = get_projection(dof, b_list[dof], Xi, projection=projection)

    Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    Pamy.PressureInitialization()
    angle_initial_read =np.array(frontend.latest().get_positions())
# %% Model
if learning_mode == 'b':
    root_model = root + '/' + 'model'
    mkdir(root_model)
    root_file = root_model + '/' + 'b_list' 
    file = open(root_file, 'wb')
    pickle.dump(b_list, file, -1)
    file.close()