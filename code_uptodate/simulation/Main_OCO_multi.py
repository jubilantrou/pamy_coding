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
# %% connect to the robot
frontend = o80_pam.FrontEnd("real_robot")  # connect to the real robot
Pamy     = PAMY_CONFIG.build_pamy(frontend=frontend)
# %% constant
# alpha_list        = [1e-8, 1e-8, 1e-9]  # not bad
# epsilon_list      = [1e-13, 0.0, 0.0]
alpha_list        = [1e-9, 1e-9, 1e-9]  # good 
epsilon_list      = [1e-13, 0.0, 0.0]
projection        = 'osqp'
learning_mode     = 'b'
mode_name         = 'ff'
step_size_version = 'sqrt'
coupling          = 'yes'
nr_channel        = 1
nr_epoch          = 8
h                 = 100
train_index       = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
test_index        = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]

if coupling == 'no':
    sum_1_list = [np.zeros((nr_channel*(h+h+1)+1, nr_channel*(h+h+1)+1))] * 3
    sum_2_list = [np.zeros((nr_channel*(h+h+1)+1, nr_channel*(h+h+1)+1))] * 3
elif coupling == 'yes':
    sum_1_list = [np.zeros((3*(h+h+1)+1, 3*(h+h+1)+1))] * 3
    sum_2_list = [np.zeros((3*(h+h+1)+1, 3*(h+h+1)+1))] * 3
# %% build folders
root          = '/home/hao/Desktop/Learning'
folder_name       = learning_mode + '_' + mode_name  + '_' + step_size_version + '_' + projection + '_' + coupling + '_' + str(nr_channel) + 'channel'
root_data     = root + '/' + 'data' + '/' + 'oco_multi' + '/' + folder_name
root_learning = root_data + '/' + 'learning'
root_verify   = root_data + '/' + 'verify'
root_model    = root_data + '/' + 'model'
mkdir(root_data)
mkdir(root_learning)
mkdir(root_verify)
mkdir(root_model)
# %% functions
def get_initial_guess(Pamys=None):
    b_list = []
    for dof in range(3):
        part1 = np.copy(Pamys[0].hessian_list[dof])
        part2 = np.copy(Pamys[0].part2_list[dof])
        b     = np.linalg.pinv(part1)@part2
        b_list.append(b)
    return b_list

def get_osqp(dof, Xi, b):
    limit_max = PAMY_CONFIG.pressure_max
    limit_min = PAMY_CONFIG.pressure_min
    
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

def get_update(b, sk, hessian, gradient, Xi):
    print('dof{}: max b {}, max update {}'.format(dof, np.max(abs(b)), np.max(abs(sk * np.linalg.pinv(hessian) @ gradient))))
    print('dof{}: max u update {}'.format(dof, np.max(abs(sk * Xi@np.linalg.pinv(hessian)@gradient))))
    b = b - sk * np.linalg.pinv(hessian) @ gradient
    return b

def get_projection(dof, b, Xi, projection='osqp'):
    if projection == 'cutoff':
        b = np.linalg.pinv(Xi)@LC.LimitCheck(Xi@b, dof)
    elif projection == 'osqp':
        b = get_osqp(dof, Xi, b)
    return b

def get_newton_method(nr, dof, robot, sum_1_list, sum_2_list, y, alpha_list, epsilon_list):
    alpha    = alpha_list[dof]
    epsilon  = epsilon_list[dof]
    sum_1    = sum_1_list[dof]
    sum_2    = sum_2_list[dof]
    L        = robot.gradient_list[dof]
    
    sum_1   += L.T@L
    Xi       = robot.Xi_list[dof]
    sum_2   += Xi.T@Xi
    hessian  = get_hessian(sum_1/nr, sum_2/nr, alpha=alpha, epsilon=epsilon)
    gradient = L.T@(y[dof, :].reshape(-1, 1) - robot.y_desired[dof, :].reshape(-1, 1)) 
    return (hessian, gradient, sum_1, sum_2, Xi)

def get_step_size(nr, dof, step_size_version='constant'):
    factor = [2.0, 2.0, 2.0]
    if step_size_version == 'constant':
        step_size = 1.5
    elif step_size_version == 'sqrt':
        step_size = factor[dof]/(2+np.sqrt(nr))
        if dof == 1:
            step_size = 0.9
    return step_size 

def get_index(index_list, index_used):
    index_rest = list(set(index_list) - set(index_used))
    l          = len(index_rest)
    index      = index_rest[random.randint(0, l-1)]
    return index

def get_verify(Pamy, b_list, path, index, mode_name='ff'):
    angle_initial_read =np.array(frontend.latest().get_positions())
    (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, mode_name=mode_name, coupling=coupling)

    root_file = path + '/' + str(index) 
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
    

def verify(Pamys, index_list, path, name='train', b_list=None):
    root_verify = path + '/' + name
    mkdir(root_verify)
    root_ff = root_verify + '/' + 'ff'
    mkdir(root_ff)
    root_fb = root_verify + '/' + 'fb'
    mkdir(root_fb)

    for i in range(len(Pamys)):
        Pamy    = Pamys[i]
        get_verify(Pamy, b_list, root_ff, index=index_list[i], mode_name='ff')
        get_verify(Pamy, b_list, root_fb, index=index_list[i], mode_name='ff+fb')
        

# def get_regret(nr, dof, y, robot, regret):
#     if  nr == 0:
#         regret[dof, nr] = 1/2 * np.linalg.norm(y[dof, :] - robot.y_desired[dof, :])
#     else:
#         regret[dof, nr] = regret[dof, nr-1] + 1/2 * np.linalg.norm(y[dof, :] - robot.y_desired[dof, :])
#     return regret            

# %% initilization
Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
# %% build Pamys for training and testing
Pamy_train = []
Pamy_test  = []

for index in train_index:
    Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
    (t, p) = PAMY_CONFIG.get_trajectory(index, mode='train')
    Pamy.ImportTrajectory(p, t)  #  import the desired trajectories and the time stamp
    Pamy.GetOptimizer_convex(angle_initial_read, coupling=coupling)
    Pamy_train.append(Pamy)

for index in test_index:
    Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
    (t, p) = PAMY_CONFIG.get_trajectory(index, mode='test')
    Pamy.ImportTrajectory(p, t)
    Pamy.GetOptimizer_convex(angle_initial_read, coupling=coupling)
    Pamy_test.append(Pamy)

# %% generate constraint
Xi_total = [None] * 3
for i in range(len(Pamy_train)):
    Pamy = Pamy_train[i]
    for dof in range(3):
        if i == 0:
            Xi_total[dof] = np.copy(Pamy.Xi_list[dof])
        else:
            Xi_total[dof] = np.concatenate((Xi_total[dof], Pamy.Xi_list[dof]), axis=0)
# %% Learning
index_used = []
i_epoch = 0

# regret    = np.zeros((3, number_iteration))
# regret_1  = np.zeros((3, number_iteration))
# regret_2  = np.zeros((3, number_iteration))

for i_epoch in range(nr_epoch):
    root_epoch = root_learning + '/' + str(i_epoch)
    mkdir(root_epoch) 

    for i_it in range(len(train_index)):
        index = get_index(train_index, index_used)
        index_used.append(index)

        Pamy    = Pamy_train[train_index.index(index)]  # find the position of index in train_index
        t_stamp = Pamy.t_stamp

        if (i_epoch == 0) and (i_it == 0):
            b_list = get_initial_guess([Pamy])
            for ii in range(4):
                (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, coupling=coupling)
                y_out = y - y[:, 0].reshape(-1, 1)
                pre_round = ii + 1
                for dof in range(len(b_list)):  # update the linear model b
                    '''
                    b = b - s_k * pinv(1/t*sum(L.T * L)+alpha/t*sum(X.T * X)+epsilon*I) * L.T * (y_out - y_des)
                    '''
                    [hessian, gradient, sum_1_list[dof], sum_2_list[dof], Xi] = get_newton_method(pre_round, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list)
                    sk                                                        = get_step_size(pre_round, dof, step_size_version=step_size_version)
                    b_list[dof]                                               = get_update(b_list[dof], sk, hessian, gradient, Xi)
                    b_list[dof]                                               = get_projection(dof, b_list[dof], Xi_total[dof], projection=projection)

                Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
                Pamy.PressureInitialization()
                angle_initial_read =np.array(frontend.latest().get_positions())
            
            # for dof in range(len(b_list)):
            #     Xi_total[dof] = np.copy(Pamy.Xi_list[dof])

        (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, coupling=coupling)
        y_out = y - y[:, 0].reshape(-1, 1)

        root_file = root_epoch + '/' + str(i_it) 
        file = open(root_file, 'wb')
        pickle.dump(t_stamp, file, -1)
        pickle.dump(t_stamp, file, -1)
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

        nr_round = i_epoch * len(train_index) + i_it + 1
        for dof in range(len(b_list)):  # update the linear model b
            '''
            b = b - s_k * pinv(1/t*sum(L.T * L)+alpha/t*sum(X.T * X)+epsilon*I) * L.T * (y_out - y_des)
            '''
            [hessian, gradient, sum_1_list[dof], sum_2_list[dof], Xi] = get_newton_method(nr_round, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list)
            # if i_epoch == 0:
            #     Xi_total[dof]                                             = np.concatenate((Xi_total[dof], Xi), axis=0)
            sk                                                        = get_step_size(nr_round, dof, step_size_version=step_size_version)
            b_list[dof]                                               = get_update(b_list[dof], sk, hessian, gradient, Xi)
            b_list[dof]                                               = get_projection(dof, b_list[dof], Xi_total[dof], projection=projection)
            # regret_1                                                  = get_regret(it, dof, y_out, Pamy, regret_1)

        Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
        Pamy.PressureInitialization()
        angle_initial_read =np.array(frontend.latest().get_positions())
    
    index_used = [] 
    root_file = root_model + '/' + str(i_epoch)  # save the model at each epoch 
    file = open(root_file, 'wb')
    pickle.dump(b_list, file, -1)
    file.close()
# %% Regret
# root_regret = root + '/' + 'regret'
# mkdir(root_regret)
# root_file = root_regret + '/' + 'regret_1' 
# file = open(root_file, 'wb')
# pickle.dump(regret_1, file, -1)
# file.close()
# %% Calculate regret
# for it in range(len(index_his)):
#     index = index_his[it]
#     Pamy  = Pamy_train[index]
#     (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, mode_name='ff+fb', 
#                                                                     coupling=coupling, 
#                                                                     learning_mode=learning_mode)
#     y_out = y - y[:, 0].reshape(-1, 1)

#     for dof in range(len(b_list)):
#         regret_2 = get_regret(it, dof, y_out, Pamy, regret_2)

#     Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
#     Pamy.PressureInitialization()
#     angle_initial_read =np.array(frontend.latest().get_positions())
# regret = regret_1 - regret_2

# root_file = root_regret + '/' + 'regret_2'
# file = open(root_file, 'wb')
# pickle.dump(regret_2, file, -1) # time stamp for x-axis
# file.close()
# root_file = root_regret + '/' + 'regret'
# file = open(root_file, 'wb')
# pickle.dump(regret, file, -1) # time stamp for x-axis
# file.close()

# xx = range(len(index_his))
# plt.plot(xx, regret[0, :], 'ro', xx, regret[1, :], 'bs', xx, regret[2, :], 'g^')
# plt.show()
# %% Verify
verify(Pamy_train, train_index, path=root_verify, name='train', b_list=b_list)
verify(Pamy_test, test_index, path=root_verify, name='test', b_list=b_list)