'''
This script is used to train the robot with online convex optimization.
control diagram description: only feedforward
ff control policy description: CNN + online gradient descent
training data description: multiple trajectories with the same initial state
'''
# %% import
import PAMY_CONFIG
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import random
from get_handle import get_handle
import o80_pam
import torch
from CNN import CNN
import torch.nn as nn
import time
from RealRobotGeometry import RobotGeometry
import time
import torch.nn as nn
# %% parameter configuration file (can be organized as an independent file)
obj = 'sim'
# parameters about the simulator setting in get_handle.py
# %% initialize the gpu and the robot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
else:
    raise ValueError('The variable obj needs to be assigned either as sim or as real!')

Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
RG = RobotGeometry()
# sigma = np.zeros(4)
# for dof in range(4):
#     sigma[dof] = np.max([PAMY_CONFIG.pressure_ago_ub[dof], np.abs(PAMY_CONFIG.pressure_ago_lb[dof])])
# max_val = PAMY_CONFIG.pressure_ago_ub / sigma 
# min_val = PAMY_CONFIG.pressure_ago_lb / sigma 
# %% constant
# alpha_list        = [1e-5, 1e-5, 1e-5]
# epsilon_list      = [1e-17, 1e-17, 1e-17]
# learning_mode     = 'b'
# mode_name         = 'ff'
# step_size_version = 'constant'
coupling          = 'yes'
nr_epoch          = 10
nr_channel        = 1
h                 = 100
nr_iteration      = 10
# version           = 'random'
# train_index       = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
# test_index        = [30, 39] #, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
# root              = '/home/hao/Desktop/MPI/Pamy_simulation'
# folder_name       = version + '_' + 'ep' + '_' + str(nr_epoch) + '_' + 'h' + '_' + str(h) + '_' + 'st' + '_' + step_size_version + '_' + str(len(train_index))
# root_data         = root + '/' + 'data' + '/' + 'oco_multi' + '/' + 'cnn_fc' + '/'  + folder_name
# root_learning     = root_data + '/' + 'learning'
# root_verify       = root_data + '/' + 'verify'
# root_model        = root_data + '/' + 'model'

# def mkdir(path):
#     folder = os.path.exists(path)
#     if not folder:
#         os.makedirs(path)

# mkdir(root_data)
# mkdir(root_learning)
# mkdir(root_verify)
# mkdir(root_model)

# gamma       = 0.3
width      = 3
ds          = 1
height       = int((2*h)/ds)+1
filter_size = 21
# learning_rate = [1e-4, 1e-5, 3e-5]
# weight_decay = 0.00
# %% functions
def get_random():
    theta = np.zeros(3)
    theta[0] = random.randrange(-300, 300)/10
    theta[1] = random.randrange(400, 700)/10
    theta[2] = random.randrange(400, 700)/10
    t        = random.randrange(90, 100)/100
    theta    = theta * math.pi/180
    return (t, theta)


def get_compensated_data(data=None):
    I_left = np.tile(data[:, 0].reshape(-1, 1), (1, h))
    I_right = np.tile(data[:, -1].reshape(-1, 1), (1, h))
    y_ = np.hstack((I_left, data, I_right))
    return y_    

def get_dataset(y, batch_size=1):
    l = y.shape[1]
    y_ = get_compensated_data(y)
    data = []
    dataset = []
    for k in range(l):
        y_temp = np.concatenate((y_[0, k:k+2*h+1:ds], y_[1, k:k+2*h+1:ds], y_[2, k:k+2*h+1:ds]))
        '''
        batchsize x channel x height x width
        '''
        data.append(torch.tensor(y_temp, dtype=float).view(nr_channel, height, width).to(device))

    idx = 0
    while idx + batch_size - 1 < l:
        data_ = data[idx:idx+batch_size]
        batch = torch.stack(data_)
        dataset.append(batch)
        idx += batch_size
    return dataset

def get_grads_list(dataset, cnn_list):
    X_list = []
    dof = 0
    for cnn in cnn_list:
        for name, param in cnn.named_parameters():
            if param.grad is None:
                break
            param.grad.zero_()
        flag = True
        for data in dataset:
            grad = []
            try:
                cnn(data.float()).mean().backward()
            except:
                cnn(data).mean().backward()
            
            
            for name, param in cnn.named_parameters():
                grad.append(torch.clone(param.grad.cpu().view(-1)))
                param.grad.zero_()
            
            # [grad.append(param.grad.view(-1)) for name, param in cnn.named_parameters()]
            grads = torch.cat(grad)
            grads_ = np.copy(grads.reshape(1, -1)) if flag else np.concatenate((grads_, grads.reshape(1, -1)), axis=0)
            flag = False if flag else False
        X_list.append(grads_)
        dof += 1
    return X_list

def get_matrix(X_list, Bu_list):
    hessian_list = []
    L_list = []
    for dof in range(3):
        gradient = Bu_list[dof]@X_list[dof] * sigma[dof]
        hessian = gradient.T@gradient
        hessian_list.append(hessian)
        L_list.append(gradient)
    return (hessian_list, L_list)

# def get_initial_guess(Pamys, cnn_list):
#     W_list = []
#     y = Pamys.y_desired
#     dataset = get_dataset(y)
#     X_list = get_grads_list(dataset, cnn_list)
#     (hessian_list, gradient_list) = get_matrix(X_list, [Pamys.O_list[dof].Bu for dof in range(3)])
#     for dof in range(3):
#         W = np.linalg.pinv(hessian_list[dof])@gradient_list[dof].T@y[dof, :].reshape(-1, 1)
#         W_list.append(W)
#     return W_list

def get_initial_guess(Pamys, cnn_list):
    W_list = [None] * 3
    avg_loss_his = [None] * 3
    nr_epoch = 50
    y = Pamys.y_desired
    dataset = get_dataset(y)
    u_list = [None] * 3

    for dof in range(3):
        u = Pamys.Xi_list[dof]@np.linalg.pinv(Pamys.hessian_list[dof])@Pamys.part2_list[dof]
        u_list[dof] = u
        u = u.flatten()/sigma[dof]
        cnn = cnn_list[dof]
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate[dof], weight_decay=weight_decay)
        loss_function = nn.MSELoss()
        # loss_function = nn.L1Loss(size_average=True)
        train_loader = []
        [train_loader.append((dataset[i], torch.tensor(u[i], dtype=float).to(device))) for i in range(len(u))]
        avg_loss_epoch = []
        for epoch in range(nr_epoch):
            avg_loss = 0.0
            cnn.train()

            for (data, label) in train_loader:
                output = cnn(data.float())
                loss = loss_function(output.squeeze(), label.squeeze().float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
        
            avg_loss /= len(train_loader)
            avg_loss_epoch.append(avg_loss)
        avg_loss_his[dof] = avg_loss_epoch
        W = []
        [W.append(param.data.view(-1)) for param in cnn.parameters()]
        W = torch.cat(W)
        W_list[dof] = W.cpu().numpy().reshape(-1, 1)
    
    fig, axs = plt.subplots(3, 1, figsize=(40, 20))
    for dof in range(3):
        line = []
        ax = axs[dof]
        ax.set_xlabel(r'Epoch')
        ax.set_ylabel(r'Loss')
        line_temp, = ax.plot(range(nr_epoch), avg_loss_his[dof], label=r'loss, dof: {}'.format(dof))
        line.append(line_temp)
        ax.grid()
        ax.legend(handles=line, loc='upper right')
    plt.show()

    return (W_list, u_list)

# def get_osqp(dof, Xi, b):    
#     m = osqp.OSQP()
#     (r, nr) = Xi.shape
#     I = np.eye(nr)
#     l = np.ones(r) * min_val[dof]
#     u = np.ones(r) * max_val[dof]

#     # q = -(Xi.T@Xi@b).flatten()
#     # A = sparse.csc_matrix(Xi)
#     # P = sparse.csc_matrix(Xi.T@Xi)
#     q = -(b).flatten()
#     A = sparse.csc_matrix(Xi)
#     P = sparse.csc_matrix(I)
#     '''
#     minimize        0.5 x' P x + q' x
#     subject to      l <= A x <= u
#     '''
#     m.setup(P=P, q=q, A=A, l=l, u=u)
#     results = m.solve()
#     return results.x.reshape(-1, 1)

# def get_projection(dof, b, Xi):
#     b = get_osqp(dof, Xi, b)
#     return b

def get_hessian(sum_1, sum_2, alpha, epsilon):
    (l, r) = sum_1.shape
    I = np.eye(l)
    hessian = sum_1 + alpha*sum_2 + epsilon*I
    return hessian

def get_update(b, sk, hessian, gradient):
    t1 = time.time()
    b = b - sk * (torch.linalg.pinv(torch.tensor(hessian))).numpy()@gradient
    print(time.time()-t1)
    return b

def get_newton_method(nr, dof, robot, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list, cnn_list):
    alpha    = alpha_list[dof]
    epsilon  = epsilon_list[dof]
    sum_1_pre = sum_1_list[dof]
    sum_2_pre = sum_2_list[dof]

    y = robot.y_desired
    dataset = get_dataset(y)
    t1 = time.time()
    X_list = get_grads_list(dataset, cnn_list)
    print(time.time()-t1)
    (hessian_list, L_list) = get_matrix(X_list, [robot.O_list[dof].Bu for dof in range(3)])

    sum_1 = gamma*sum_1_pre + (1-gamma)*hessian_list[dof]  # L.T@L
    sum_2 = gamma*sum_2_pre + (1-gamma)*X_list[dof].T@X_list[dof]*sigma[dof]*sigma[dof]

    hessian  = get_hessian(sum_1, sum_2, alpha=alpha, epsilon=epsilon)
    gradient = L_list[dof].T@(y_out[dof, :].reshape(-1, 1) - robot.y_desired[dof, :].reshape(-1, 1)) 
    return (hessian, gradient, sum_1, sum_2, X_list)

def get_step_size(nr, dof, step_size_version='constant'):
    factor = [1.0, 1.0, 1.0]
    # constant_list = [0.2, 0.2, 0.5]
    constant_list = [3e-1, 3e-1, 3e-1]
    if step_size_version == 'constant':
        step_size = constant_list[dof]
    elif step_size_version == 'sqrt':
        step_size = factor[dof]/(2+np.sqrt(nr))
        # if dof == 1:
        #     step_size = 2.0
    return step_size 

# def get_index(index_list, index_used):
#     index_rest = list(set(index_list) - set(index_used))
#     l          = len(index_rest)
#     index      = index_rest[random.randint(0, l-1)]
#     return index

# def get_verify(Pamy, b_list, path, index, mode_name='ff'):
#     angle_initial_read =np.array(frontend.latest().get_positions())
#     (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(b_list, mode_name=mode_name, coupling=coupling)

#     root_file = path + '/' + str(index) 
#     file = open(root_file, 'wb')
#     pickle.dump(t_stamp, file, -1) # time stamp for x-axis
#     pickle.dump(t, file, -1)
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
#     Pamy.PressureInitialization()
    
# def verify(Pamys, index_list, path, name='train', b_list=None):
#     root_verify = path + '/' + name
#     mkdir(root_verify)
#     root_ff = root_verify + '/' + 'ff'
#     mkdir(root_ff)
#     root_fb = root_verify + '/' + 'fb'
#     mkdir(root_fb)

#     for i in range(len(Pamys)):
#         Pamy    = Pamys[i]
#         get_verify(Pamy, b_list, root_ff, index=index_list[i], mode_name='ff')
#         get_verify(Pamy, b_list, root_fb, index=index_list[i], mode_name='ff+fb')

def set_parameters(W_list, cnn_list, idx_list, shape_list):
    for dof in range(3):
        W = W_list[dof]
        cnn = cnn_list[dof]
        i = 0
        for name, param in cnn.named_parameters():
            idx_1 = idx_list[i]
            idx_2 = idx_list[i+1]
            W_ = torch.tensor(W[idx_1:idx_2]).view(shape_list[i])
            param.data = W_.to(device)
            i += 1
    return cnn_list

def get_prediction(cnn_list, y, denorm):
    dataset = get_dataset(y, batch_size=y.shape[1])
    u = np.zeros(y.shape)
    for dof in range(3):
        cnn = cnn_list[dof]
        cnn.eval()
        try:
            u[dof, :] = cnn(dataset[0]).cpu().detach().numpy().flatten() * denorm[dof]
        except:
            u[dof, :] = cnn(dataset[0].float()).cpu().detach().numpy().flatten() * denorm[dof]
    return u
# %% initilization
print('init')
Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
# Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())
# %% define the cnn
cnn_list   = []
name_list  = []
shape_list = []
idx_list   = []
idx = 0
idx_list.append(idx)

def weight_init(l):
    if isinstance(l,nn.Conv2d) or isinstance(l,nn.Linear):
        nn.init.constant_(l.weight,0.0)
        nn.init.constant_(l.bias,0.0)

for dof in range(3):
    cnn = CNN(filter_size=filter_size, width=width, height=height, channel_in=nr_channel)
    cnn.apply(weight_init)
    cnn.to(device)
    cnn_list.append(cnn)

for name, param in cnn.named_parameters():  # models are the same for all dofs
    name_list.append(name)
    shape_list.append(param.shape)
    d_idx = len(param.data.view(-1))
    idx += d_idx
    idx_list.append(idx)
print(idx_list)
# %% build Pamys for training and testing
# Pamy_train = []
# Pamy_test  = []

# for index in train_index:
#     Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
#     (t, p) = PAMY_CONFIG.get_trajectory(index, mode='train')
#     Pamy.ImportTrajectory(p, t)  #  import the desired trajectories and the time stamp
#     Pamy.GetOptimizer_convex(angle_initial_read, nr_channel=nr_channel, coupling=coupling)
#     Pamy_train.append(Pamy)

# for index in test_index:
#     Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
#     (t, p) = PAMY_CONFIG.get_trajectory(index, mode='test')
#     Pamy.ImportTrajectory(p, t)
#     Pamy.GetOptimizer_convex(angle_initial_read, nr_channel=nr_channel, coupling=coupling)
#     Pamy_test.append(Pamy)
# %% Learning
index_used = []

(t, angle) = get_random()
print(angle/math.pi*180)

for i_epoch in range(nr_epoch):
    # root_epoch = root_learning + '/' + str(i_epoch)
    # mkdir(root_epoch) 

    # if i_epoch == 0:  # initialization
    #     Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
    #     (t, angle) = get_random()
    #     (p, v, a, j, theta, t_stamp) = RG.PathPlanning(angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)
    #     theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    #     theta = theta - theta[:, 0].reshape(-1, 1)
    #     Pamy.ImportTrajectory(theta, t_stamp)  #  import the desired trajectories and the time stamp
    #     Pamy.GetOptimizer_convex(angle_initial_read, nr_channel=nr_channel, coupling=coupling)

    #     # index   = get_index(train_index, index_used)
    #     # index_used.append(index)
    #     # Pamy    = Pamy_train[train_index.index(index)]  # find the position of index in train_index
    #     # t_stamp = Pamy.t_stamp

    #     (W_list, u_ini) = get_initial_guess(Pamy, cnn_list)
    #     sum_1_list = [np.zeros((len(W_list[0]), len(W_list[0])))] * 3
    #     sum_2_list = [np.zeros((len(W_list[0]), len(W_list[0])))] * 3

    #     u = get_prediction(cnn_list, Pamy.y_desired)

    #     fig, axs = plt.subplots(3, 1, figsize=(40, 20))
    #     for dof in range(3):
    #         line = []
    #         ax = axs[dof]
    #         ax.set_xlabel(r'Time $t$ in s')
    #         ax.set_ylabel(r'Normalized input')
    #         line_temp, = ax.plot(t_stamp, u[dof, :].flatten(), label=r'prediction')
    #         line.append(line_temp)
    #         line_temp, = ax.plot(t_stamp, u_ini[dof].flatten(), label=r'from linear model')
    #         line.append(line_temp)
    #         ax.grid()
    #         ax.legend(handles=line, loc='upper right')
    #     plt.show()

    #     for nr_round in range(1, 6):
    #         u = get_prediction(cnn_list, Pamy.y_desired)
    #         (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(u, coupling=coupling, learning_mode='u')
    #         y_out = y - y[:, 0].reshape(-1, 1)

    #         for dof in range(3):  
    #             [hessian, gradient, sum_1_list[dof], sum_2_list[dof], X_list] = get_newton_method(nr_round, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list, cnn_list)
    #             sk                                                            = get_step_size(nr_round, dof, step_size_version=step_size_version)
    #             W_list[dof]                                                   = get_update(W_list[dof], sk, hessian, gradient)
    #         cnn_list = set_parameters(W_list, cnn_list, idx_list, shape_list)

    #         print('initialization')
    #         Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
    #         Pamy.PressureInitialization()
    #         angle_initial_read =np.array(frontend.latest().get_positions())

    #     index_used = []
    #     # sum_1_list = [np.zeros((len(W_list[0]), len(W_list[0])))] * 3
    #     # sum_2_list = [np.zeros((len(W_list[0]), len(W_list[0])))] * 3

    # for i_it in range(len(train_index)):
    for i_it in range(nr_iteration):
        index = i_it

        # Pamy   = PAMY_CONFIG.build_pamy(frontend=frontend)
        # (t, angle) = get_random()
        (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle)
        theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
        theta = theta - theta[:, 0].reshape(-1, 1)
        Pamy.ImportTrajectory(theta, t_stamp)  #  import the desired trajectories and the time stamp
        Pamy.GetOptimizer_convex(angle_initial_read, nr_channel=nr_channel, coupling=coupling)
        
        # index = get_index(train_index, index_used)
        # index_used.append(index)

        # Pamy    = Pamy_train[train_index.index(index)]  # find the position of index in train_index
        # t_stamp = Pamy.t_stamp

        u = get_prediction(cnn_list, Pamy.y_desired, PAMY_CONFIG.pressure_limit)

        (y, ff, fb, obs_ago, obs_ant) = Pamy.online_convex_optimization(u, coupling=coupling, learning_mode='u')
        y_out = y - y[:, 0].reshape(-1, 1)

        # root_file = root_epoch + '/' + str(i_it) 
        # file = open(root_file, 'wb')
        # pickle.dump(t_stamp, file, -1)
        # pickle.dump(t_stamp, file, -1)
        # pickle.dump(angle_initial_read, file, -1)
        # pickle.dump(y, file, -1)
        # pickle.dump(Pamy.y_desired, file, -1)
        # pickle.dump(ff, file, -1)
        # pickle.dump(fb, file, -1)
        # pickle.dump(obs_ago, file, -1)
        # pickle.dump(obs_ant, file, -1)
        # pickle.dump(W_list, file, -1)
        # pickle.dump(index, file, -1)
        # file.close()

        print('begin {},{}. optimization'.format(i_epoch, i_it))
        # nr_round = i_epoch * len(train_index) + i_it + 1 + 5
        W_list = [None] * 3
        for i in range(3):    
            W = []
            [W.append(param.data.view(-1)) for param in cnn_list[i].parameters()]
            W = torch.cat(W)
            W_list[i] = W.cpu().numpy().reshape(-1, 1)
        part3 = get_grads_list(get_dataset(Pamy.y_desired), cnn_list)
        part2 = [Pamy.O_list[i].Bu for i in PAMY_CONFIG.dof_list]
        part1_temp = y_out - Pamy.y_desired
        part1 = [part1_temp[i].reshape(1,-1) for i in range(len(part1_temp))]
        loss = [np.linalg.norm(part1_temp[i].reshape(1,-1)) for i in range(len(part1_temp))]
        step_list = [0.5, 0.5, 0.5]
        for dof in range(3):  # update the linear model b
            W_list[dof] = W_list[dof] - step_list[dof]*(part1[dof]@part2[dof]@part3[dof]).reshape(-1, 1)
        #     '''
        #     b = b - s_k * pinv(1/t*sum(L.T * L)+alpha/t*sum(X.T * X)+epsilon*I) * L.T * (y_out - y_des)
        #     '''

        #     [hessian, gradient, sum_1_list[dof], sum_2_list[dof], X_list] = get_newton_method(nr_round, dof, Pamy, sum_1_list, sum_2_list, y_out, alpha_list, epsilon_list, cnn_list)
        #     sk                                                            = get_step_size(nr_round, dof, step_size_version=step_size_version)
        #     W_list[dof]                                                   = get_update(W_list[dof], sk, hessian, gradient)
        #     # W_list[dof]                                                   = get_projection(dof, W_list[dof], X_list[dof])
        cnn_list = set_parameters(W_list, cnn_list, idx_list, shape_list)
        print('loss:')
        print(loss)

        print('____________')
        print('initialization for next training')
        Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
        # Pamy.PressureInitialization()
        angle_initial_read =np.array(frontend.latest().get_positions())
    
    # index_used = [] 
    # root_model_epoch = root_model + '/' + str(i_epoch)  # save the model at each epoch
    # mkdir(root_model_epoch) 
    # for dof in range(3):
    #     cnn = cnn_list[dof]
    #     root_file = root_model_epoch + '/' + str(dof)
    #     torch.save(cnn.state_dict(), root_file)
# %% Verify
# verify(Pamy_train, train_index, path=root_verify, name='train', b_list=b_list)
# verify(Pamy_test, test_index, path=root_verify, name='test', b_list=b_list)