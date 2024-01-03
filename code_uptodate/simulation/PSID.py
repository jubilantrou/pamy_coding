'''
This script is for pseudo system identification.
'''
import math
import numpy as np
import time
import o80
import o80_pam
import matplotlib.pyplot as plt
import os
import PAMY_CONFIG
import pickle5 as pickle
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
# %% constant
T                 = 10  # in second
freq              = 100  # in Hz
dt                = 1/freq
# anchor_ago_list   = np.array([17500, 18500, 16000, 8000])
# anchor_ant_list   = np.array([17500, 18500, 16000, 8000])

anchor_ago_list   = np.array([17500, 20000, 16000, 8000])
anchor_ant_list   = np.array([17500, 17000, 16000, 8000])

frequency_backend = 500.0  # frequency of the backend 
ipc               = int(frequency_backend/freq)  # sychronize the frontend and the backend
frontend          = o80_pam.FrontEnd("real_robot")
Geometry          = PAMY_CONFIG.build_geometry()
Pamy              = PAMY_CONFIG.build_pamy(frontend=frontend)
# initial_posture   = np.array([0.0, 0.0, 0.0, 0.0])
initial_posture   = Geometry.initial_posture
# %% read data from files
root          = '/home/hao/Desktop/Learning/data'
root_response = root + '/' + 'psid_response'
root_input    = root + '/' + 'pseudo_sid'
mkdir(root_response)
u_list  = []
files   = os.listdir(root_input)
nr      = len(files)
for idx in range(nr):
    root_file = root_input + '/' + str(idx)
    f = open(root_file, 'rb')
    t = pickle.load(f)
    u = pickle.load(f)
    f.close()
    t_stamp = t[0:T*freq]
    u_list.append(u[:, 0:T*freq])
# %% run the system
for i in range(len(u_list)):
    u        = u_list[i]
    u[1, :]  = -u[1, :] * 1.5
    u[2, :]  = -u[2, :] * 1.5
    position = np.zeros(u.shape)

    Pamy.AngleInitialization(initial_posture)
    Pamy.PressureInitialization()

    ref_iteration    = frontend.latest().get_iteration()  # read the current iteration
    iteration_begin  = ref_iteration + 1000  # set the iteration when beginning
    iteration        = iteration_begin
    frontend.add_command(anchor_ago_list, anchor_ant_list,
                         o80.Iteration(iteration_begin-ipc),
                         o80.Mode.QUEUE)
    frontend.pulse()

    print("begin the system identification, index.{}".format(i))
    for idx in range(u.shape[1]):
        diff = u[:, idx]
        pressure_ago = (anchor_ago_list + diff).astype(int)  # red line --> agonist
        pressure_ant = (anchor_ant_list - diff).astype(int)  # green line --> antagonist
        
        frontend.add_command(pressure_ago, pressure_ant,
                             o80.Iteration(iteration),
                             o80.Mode.QUEUE)
        frontend.add_command(pressure_ago, pressure_ant,
                             o80.Iteration(iteration + ipc - 1),
                             o80.Mode.QUEUE)
        frontend.pulse()

        iteration += ipc

    iteration_end  = iteration
    iteration      = iteration_begin
    pointer        = 0
    while iteration < iteration_end:
        observation          = frontend.read(iteration)
        position[:, pointer] = observation.get_positions()
        iteration           += ipc
        pointer             += 1
    
    root_file = root_response + '/' + str(i)
    file = open(root_file, 'wb')
    pickle.dump(t_stamp,  file, -1)
    pickle.dump(u,        file, -1)
    pickle.dump(position, file, -1)
    file.close()