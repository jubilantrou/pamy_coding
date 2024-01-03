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
from get_handle import get_handle
from RealRobotGeometry import RobotGeometry
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
# %% connect to the robot
handle   = get_handle(mode='pressure')
frontend = handle.frontends["robot"]
Pamy     = PAMY_CONFIG.build_pamy(frontend=frontend) 
RG       = RobotGeometry()
# %% constant
alpha_list        = [1e-7, 1e-9, 1e-9]
epsilon_list      = [1e-13, 1e-17, 1e-15]
projection        = 'osqp'
learning_mode     = 'b'
mode_name         = 'ff'
step_size_version = 'sqrt'
coupling          = 'yes'
nr_epoch          = 5
nr_iteration      = 30
nr_channel        = 1
h                 = 100
if coupling == 'no':
    sum_1_list        = [np.zeros((nr_channel*(h+h+1)+1, nr_channel*(h+h+1)+1))] * 3
    sum_2_list        = [np.zeros((nr_channel*(h+h+1)+1, nr_channel*(h+h+1)+1))] * 3
elif coupling == 'yes':
    sum_1_list        = [np.zeros((3*(h+h+1)+1, 3*(h+h+1)+1))] * 3
    sum_2_list        = [np.zeros((3*(h+h+1)+1, 3*(h+h+1)+1))] * 3
root              = '/home/hao/Desktop/MPI/Pamy_simulation'
folder_name       = 'random' + '_' + learning_mode + '_' + mode_name  + '_' + step_size_version + '_' + projection + '_' + coupling + '_' + str(nr_channel) + 'channel'
root_data         = root + '/' + 'data' + '/' + 'oco_multi' + '/' + folder_name
root_learning     = root_data + '/' + 'learning'
root_verify       = root_data + '/' + 'verify'
root_model        = root_data + '/' + 'model'
root_constraint   = root_data + '/' + 'constraint'
mkdir(root_data)
mkdir(root_learning)
mkdir(root_verify)
mkdir(root_model)
mkdir(root_constraint)
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
    limit_max = PAMY_CONFIG.ago_max_list - PAMY_CONFIG.anchor_ago_list
    limit_min = PAMY_CONFIG.ago_min_list - PAMY_CONFIG.anchor_ago_list
    
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
    b = b - sk * np.linalg.pinv(hessian)@gradient
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
        step_size = 1.0
    elif step_size_version == 'sqrt':
        step_size = factor[dof]/(2+np.sqrt(nr))
        # if dof == 1:
        #     step_size = 2.0
    return step_size 

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

# t:      [0.85, 1.0]
# theta0: [-15, 15]
# tehta1: [50, 80]
# theta2: [30, 80]
def get_random():
    theta = np.zeros(3)
    theta[0] = random.randrange(-150, 150)/10
    theta[1] = random.randrange(500, 800)/10
    theta[2] = random.randrange(300, 700)/10
    t     = random.randrange(85, 100)/100
    theta = theta * math.pi/180
    return (t, theta)

def get_Xi(y):
    def get_compensated_data(data=None):
        I_left = np.tile(data[:, 0].reshape(-1, 1), (1, h))
        I_right = np.tile(data[:, -1].reshape(-1, 1), (1, h))
        new_data = np.hstack((I_left, data, I_right))
        return new_data
    y_comp = get_compensated_data(y[0:3, :])
    Xi = np.zeros((y.shape[1], 3*(h+h+1)+1))
    for k in range(y.shape[1]):
        Xi[k, :] = np.hstack((y_comp[0, k:k+2*h+1], y_comp[1, k:k+2*h+1], y_comp[2, k:k+2*h+1], 1))
    return Xi

def get_peak(y):
    t = []
    for i in range(y.shape[0]):
        t1 = np.where(np.abs(y[i, :]) == np.amax(np.abs(y[i, :])))
        t.append(t1[0][0])
    return t

def get_constraint(nr_sample=1000):
    for i in range(nr_sample):
        (t, angle) = get_random()
        (p, v, a, j, theta, t_stamp) = RG.PathPlanning(angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)
        theta = theta - theta[:, 0].reshape(-1, 1)
        # RG.GetPlot(p, theta, t_list=np.array([0.0, t, t+1.5, t+1.7]))
        Xi = get_Xi(theta)
        tt = get_peak(theta)
        if i == 0:
            Cons = Xi[tt, :]
        else:
            Cons = np.concatenate((Cons, Xi[tt, :]), axis=0)
    return Cons

def get_constraint_sample(Xi_total, Xi):
    lst1 = range(Xi_total.shape[0])
    t1 = random.sample(lst1, k=round(len(lst1) * 0.8))
    t2 = list(range(0, Xi.shape[0], 2))
    Xi_ = np.concatenate((Xi_total[t1, :], Xi[t2, :]), axis=0)
    return Xi_


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
# %% Learning
index_used = []
pre_round = 1
Xi_total = [None] * 3

# Cons = get_constraint(nr_sample=10)
# root_file = root_constraint + '/' + 'cons'  # save the model at each epoch 
# file = open(root_file, 'wb')
# pickle.dump(Cons, file, -1)
# file.close()

for i_epoch in range(nr_epoch):
    root_epoch = root_learning + '/' + str(i_epoch)
    mkdir(root_epoch) 

    for i_it in range(nr_iteration):
        Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
        (t, angle) = get_random()
        (p, v, a, j, theta, t_stamp) = RG.PathPlanning(angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)
        theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
        theta = theta - theta[:, 0].reshape(-1, 1)
        Pamy.ImportTrajectory(theta, t_stamp)  #  import the desired trajectories and the time stamp
        Pamy.GetOptimizer_convex(angle_initial_read, nr_channel=nr_channel, coupling=coupling)

        if (i_epoch == 0) and (i_it == 0):
            b_list = get_initial_guess([Pamy])
            # for ii in range(5):
            #     (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, coupling=coupling)
            #     y_out = y - y[:, 0].reshape(-1, 1)
            #     pre_round = ii + 1
            #     for dof in range(len(b_list)):  # update the linear model b
            #         [hessian, gradient, sum_1_list[dof], sum_2_list[dof], Xi] = get_newton_method(pre_round, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list)
            #         sk                                                        = get_step_size(pre_round, dof, step_size_version=step_size_version)
            #         b_list[dof]                                               = get_update(b_list[dof], sk, hessian, gradient, Xi)
            #         b_list[dof]                                               = get_projection(dof, b_list[dof], Xi, projection=projection)
            #     Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
            #     Pamy.PressureInitialization()
            #     angle_initial_read =np.array(frontend.latest().get_positions())
            
            for dof in range(3):
                Xi_total[dof] = np.copy(Pamy.Xi_list[dof])

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
        pickle.dump(i_it, file, -1)
        file.close()

        nr_round = i_epoch * nr_iteration + i_it + 1
        for dof in range(len(b_list)):  # update the linear model b
            '''
            b = b - s_k * pinv(1/t*sum(L.T * L)+alpha/t*sum(X.T * X)+epsilon*I) * L.T * (y_out - y_des)
            '''
            [hessian, gradient, sum_1_list[dof], sum_2_list[dof], Xi] = get_newton_method(nr_round, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list)
            # Xi_total[dof]                                             = np.concatenate((Xi_total[dof], Xi), axis=0)
            Xi_total[dof] = get_constraint_sample(Xi_total[dof], Xi)
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
# %% Verify
# verify(Pamy_train, train_index, path=root_verify, name='train', b_list=b_list)
# verify(Pamy_test, test_index, path=root_verify, name='test', b_list=b_list)