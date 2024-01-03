'''
This script is used to plot figures for online convex optimization
'''
import numpy as np
import pickle5 as pickle
import math
import matplotlib.pyplot as plt
import os
import PAMY_CONFIG
# %% global variables
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

train_index       = [17, 62, 1, 41, 37, 32, 67, 15, 70, 64, 23, 28, 66, 33, 35, 34, 54, 58, 38, 56, 47, 55, 11, 59, 21, 4, 48, 65, 14, 52]
test_index        = [30, 39, 50, 7, 45, 53, 16, 57, 68, 61, 60, 6, 13]
base              = np.array([17500, 18500, 16000, 15000])
legend_position   = 'lower right'
dof_list          = [0,1,2]
folder_index      = '17'
projection        = 'osqp'
learning_mode     = 'b'
mode_name         = 'ff'
coupling          = 'yes'
step_size_version = 'sqrt'
nr_channel        = 1
folder_name       = 'oco_single' + '_' + learning_mode + '_' + mode_name  + '_' + step_size_version + '_' + projection + '_' + coupling + '_' + str(nr_channel) + 'channel'
root_data         = '/home/hao/Desktop/Learning/data' + '/' + folder_name + '/' + folder_index
save_mode         = 'off'
root_figure       = '/home/hao/Desktop/Learning/figure' + '/' + folder_name + '/' + folder_index
mkdir(root_figure)
dnr               = 5
# %% functions
def get_data(path=None):
    # open file and read the data
    f = open(path, 'rb')
    t_stamp       = pickle.load(f)
    t             = pickle.load(f)
    angle_initial = pickle.load(f)
    y             = pickle.load(f)
    y_des         = pickle.load(f)
    ff            = pickle.load(f)
    fb            = pickle.load(f)
    obs_ago       = pickle.load(f)
    obs_ant       = pickle.load(f)
    b_list        = pickle.load(f)
    f.close()

    y_des = y_des + angle_initial.reshape(-1, 1)
    # for i in range(ff.shape[0]):
    #     ff[i, ff[i, :]>limit_max[i]] = limit_max[i]
    #     ff[i, ff[i, :]<limit_min[i]] = limit_min[i]
    return (t_stamp, t, y, y_des, ff, fb, obs_ago, obs_ant)

def AngleToEnd(angle):
    '''angle is absolute'''
    l_1 = 0.38
    l_2 = 0.40
    theta0 = angle[0]
    theta1 = angle[1]
    theta2 = angle[2]

    x = math.cos(theta0) * math.sin(theta1) * l_1
    y = math.sin(theta0) * math.sin(theta1) * l_1
    z = math.cos(theta1) * l_1
    position_A = np.array([x, y, z])
    x = math.cos(theta0) * math.sin(theta1) * l_1 + math.cos(theta0) * math.sin(theta1 + theta2) * l_2
    y = math.sin(theta0) * math.sin(theta1) * l_1 + math.sin(theta0) * math.sin(theta1 + theta2) * l_2
    z = math.cos(theta1) * l_1 + math.cos(theta1 + theta2) * l_2
    position_B = np.array([x, y, z])

    return (position_A, position_B)

def get_traj(y):
    l = y.shape[1]
    traj = np.zeros((3, l))
    for i in range(l):
        (_, traj[:, i]) = AngleToEnd(y[:, i])
    return traj

def get_error(y, y_des):
    l          = y.shape[1]
    p          = get_traj(y)
    p_des      = get_traj(y_des)
    error_mean = np.zeros(4)
    error_max  = np.zeros(4)

    for i in range(3):
        error_mean[i] = np.linalg.norm(y[i, :]-y_des[i, :], ord=1)/l
        error_max[i]  = np.max(np.abs(y[i, :]-y_des[i, :]))
    
    error_     = 0
    for i in range(l):
        error_ = np.linalg.norm(p[:, i]-p_des[:, i], ord=2)
        error_mean[3] += error_
        if error_ > error_max[3]:
            error_max[3] = error_

    error_mean[3] = error_mean[3]/l
    return (error_mean, error_max)

def get_path(folder):
    root_data_folder    = root_data + '/' + folder
    root_figure_folder  = root_figure + '/' + folder
    mkdir(root_figure_folder)
    return (root_data_folder, root_figure_folder)

def get_plot_index(nr, dt=1):
    index_list = []
    index = 0
    while index < nr:
        index_list.append(index)
        index += dt
    if not(nr-1 in index_list):
        index_list.append(nr-1)
    return index_list

def get_plot(path, dnr):
    (data_path, figure_path) = get_path(folder=path) 
    files                    = os.listdir(data_path)  # read all the files 
    nr_iteration             = len(files)             
    error_mean               = np.zeros((len(dof_list)+1, nr_iteration))
    error_max                = np.zeros((len(dof_list)+1, nr_iteration))
    plot_index               = get_plot_index(nr_iteration, dnr)
    # plot_index = [0, 10, 20, 27, 28, 29]
    y_list       = []
    ff_list      = []
    obs_ago_list = []
    obs_ant_list = []
    
    i = 0 
    for file in range(nr_iteration):
        root_folder_file = data_path + '/' + str(file)
        (t_stamp, t, y, y_des, ff, fb, obs_ago, obs_ant) = get_data(root_folder_file)
        (error_mean[:, i], error_max[:, i]) = get_error(y, y_des)
        
        if file in plot_index:
            y_list.append(y)
            ff_list.append(ff)
            obs_ago_list.append(obs_ago)
            obs_ant_list.append(obs_ant)

        if save_mode == 'off':
            fig, axs = plt.subplots(3, len(dof_list), figsize=(40, 20))
            for i_dof in range(len(dof_list)):
                
                ax = axs[0, i_dof]
                ax.set_xlabel(r'Time $t$ in s')
                ax.set_ylabel(r'Angle $\theta$ in degree')
                line = []
                line_temp, = ax.plot(t_stamp, y_des[i_dof, :]* 180/math.pi, 
                                    linewidth=1.5, linestyle='--', label=r'des')
                line.append(line_temp)
                line_temp, = ax.plot(t_stamp, y[i_dof, :]* 180/math.pi, 
                                    linewidth=1.0, linestyle='-', label=r'res')
                line.append(line_temp)

                ax.grid()
                ax.legend(handles=line, loc=legend_position)

                ax = axs[1, i_dof]
                ax.set_xlabel(r'Time $t$ in s')
                ax.set_ylabel(r'Normalized pressure')
                line = []
                line_temp, = ax.plot(t_stamp, ff[i_dof, :], 
                                    linewidth=1.0, linestyle='-', label=r'ff')
                line.append( line_temp )
                ax.axhline(x=PAMY_CONFIG.pressure_max[i_dof], color='b')
                ax.axhline(x=PAMY_CONFIG.pressure_min[i_dof], color='b')
                ax.grid()
                ax.legend(handles=line, loc=legend_position)

                ax = axs[2, i_dof]
                ax.set_xlabel(r'Time $t$ in s')
                ax.set_ylabel(r'Normalized pressure')
                line = []
                line_temp, = ax.plot(t_stamp, obs_ago[i_dof, :], 
                                    linewidth=1.0, linestyle='-', label=r'ago')
                line.append( line_temp )
                ax.axhline(x=PAMY_CONFIG.ago_max_list[i_dof], color='b')
                ax.axhline(x=PAMY_CONFIG.ago_min_list[i_dof], color='b')          
                ax.grid()
                ax.legend(handles=line, loc=legend_position)
            # plt.show() 
            plt.savefig(figure_path + '/' + str(file) + '.pdf')
            plt.close()
        i += 1

    fig, axs = plt.subplots(3, len(dof_list), figsize=(40, 20))
    for i_dof in range(len(dof_list)):
        line = []
        ax = axs[0, i_dof]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Error $\Delta \theta$ in degree')
        line_temp, = ax.plot(t_stamp, y_des[i_dof, :]*180/math.pi, linewidth=1.5, linestyle='--', label=r'des')
        line.append(line_temp)
        for index in range(len(plot_index)):
            line_temp, = ax.plot(t_stamp, y_list[index][i_dof, :]*180/math.pi, label=r'it.{}'.format(plot_index[index]))
            line.append(line_temp)
        ax.grid()
        ax.legend(handles=line, loc=legend_position)

        line = []
        ax = axs[1, i_dof]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Normalized pressure')
        for index in range(len(plot_index)):
            line_temp, = ax.plot(t_stamp, ff_list[index][i_dof, :], label=r'it.{}'.format(plot_index[index]))
            line.append(line_temp)
        ax.grid()
        ax.legend(handles=line, loc=legend_position)

        line = []
        ax = axs[2, i_dof]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Normalized pressure')
        for index in range(len(plot_index)):
            line_temp, = ax.plot(t_stamp, obs_ago_list[index][i_dof, :], label=r'ago it.{}'.format(plot_index[index]))
            line.append(line_temp)
            line_temp, = ax.plot(t_stamp, obs_ant_list[index][i_dof, :], label=r'ant it.{}'.format(plot_index[index]))
            line.append(line_temp)
        ax.grid()
        ax.legend(handles=line, loc=legend_position)
    
    plt.savefig(figure_path + '/' + 'learning' + '.pdf')
    plt.close()
    return (error_mean, error_max, nr_iteration)
# %%
(error_mean, error_max, nr_iteration) = get_plot(path='learning', dnr=dnr)
# %%
root_file = root_figure + '/' + 'error'
mkdir(root_file)

fig, axs = plt.subplots(4, 1, figsize=(40, 20))
for i_dof in range(len(dof_list)):
    line = []
    ax = axs[i_dof]
    ax.set_xlabel(r'Iterations')
    ax.set_ylabel(r'Error $\delta \theta$ in degree')
    line_temp, = ax.plot(range(nr_iteration), error_mean[i_dof, :]*180/math.pi, label=r'dof-{}, mean'.format(i_dof))
    line.append(line_temp)
    line_temp, = ax.plot(range(nr_iteration), error_max[i_dof, :]*180/math.pi, label=r'dof-{}, max'.format(i_dof))
    line.append(line_temp)
    ax.grid()
    ax.legend(handles=line, loc=legend_position)
line = []
ax = axs[3]
ax.set_xlabel(r'Iterations')
ax.set_ylabel(r'Error $\Delta s$ in m')
line_temp, = ax.plot(range(nr_iteration), error_mean[3, :], label=r'mean')
line_temp, = ax.plot(range(nr_iteration), error_max[3, :], label=r'max')
line.append(line_temp)
ax.grid()
ax.legend(handles=line, loc=legend_position)
# plt.show()
plt.savefig(root_file + '/' + 'learning' + '.pdf')
plt.close()