'''
This script is used to do the system identification
'''
import math
from pickle import GLOBAL
import numpy as np
import time
import o80
import o80_pam
import matplotlib.pyplot as plt
import os
import PAMY_CONFIG
# %% read files
# GLOBAL = np.array([0, -20, 0, 0])*math.pi/180

pid_list = np.array([[-13000.000,  0.00, -300],
                     [-30000.000,  0.00, -300],
                     [ -5000.000, -8000, -100],
                     [34221.8733, 167366.3594, 73.2381]])

mode = 'ff+fb'

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

frontend = o80_pam.FrontEnd("real_robot")
Pamy     = PAMY_CONFIG.build_pamy(frontend=frontend)
Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
# Pamy.PressureInitialization()
hz_limit    = 4
p           = 10  # how many periods
root        = '/home/hao/Desktop/Learning/data'
root_input  = root + '/' + 'excitation_signal'
f_input     = root_input + '/' + 'excitation_' + str(hz_limit) + 'hz.csv'
root_output = root + '/' + 'response_signal' + '/' + str(hz_limit) + 'hz'
mkdir(root_output)
u           = np.loadtxt(f_input, delimiter=',')
[m, N]      = u.shape
u           = np.tile(u, p).flatten()
anchor_ago_list = PAMY_CONFIG.anchor_ago_list
anchor_ant_list = PAMY_CONFIG.anchor_ant_list
# %% set parameters
strategy               = 1
fs                     = 100
frequency              = 500.0  # frequency of the backend 
period                 = 1.0/frequency  # period of backend
duration_per_command   = 1/fs  # period of frontend
iterations_per_command = int(duration_per_command/period)  # sychronize the frontend and the backend
if mode == 'ff+fb' or mode == 'fb+ff':
    amp_list           = [math.pi/4000]  # amp_u * 1000
elif mode == 'ff':
    amp_list           = [9]  # amp_u * 1000
t_stamp_u              = np.arange(0, (m * p * N)/fs, 1/fs)
t                      = 1 / fs
T                      = m * p * N / fs

for dof in range(2, 3):
    anchor_ago             = anchor_ago_list[dof]
    anchor_ant             = anchor_ant_list[dof]
    pid = pid_list[dof]

    for amp in amp_list:
        Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
        Pamy.PressureInitialization()

        u_temp   = u * amp  #change the amplitude of the signal
        if mode == 'ff':
            f_output = root_output + '/' + "response_dof_" + str(dof) + '_amp_' + str(amp) + '.txt'
        else:
            f_output = root_output + '/' + "response_dof_" + str(dof) + '_fb' + '.txt'
        # initilization
        position         = np.array([])
        velocity         = np.array([])
        obs_pressure_ant = np.array([])
        obs_pressure_ago = np.array([])
        des_pressure_ant = np.array([])
        des_pressure_ago = np.array([])
        t_stamp          = np.array([])
        diff             = np.zeros(len(u_temp))
        theta_list       = frontend.latest().get_positions()
        theta            = theta_list[dof]
        angle_initial    = theta_list[dof]
        res_i            = 0
        angle_delta_pre  = 0

        ref_iteration    = frontend.latest().get_iteration()  # read the current iteration
        iteration_begin  = ref_iteration + 1000  # set the iteration when beginning
        iteration        = iteration_begin
        frontend.add_command(dof, anchor_ago, anchor_ant,
                            o80.Iteration(iteration_begin-iterations_per_command),
                            o80.Mode.QUEUE)
        frontend.pulse()

        print("begin to excite the mujoco...")
        for i in range(0, len(u_temp)):
            if mode == 'ff':
                diff[i] = u_temp[i]
            elif mode == 'ff+fb' or mode == 'fb+ff':
                angle_delta     = u_temp[i]+angle_initial-theta
                res_d           = (angle_delta - angle_delta_pre)/t
                res_i          += angle_delta * t
                diff[i]         = pid[0]*angle_delta + pid[1]*res_i + pid[2]*res_d
                angle_delta_pre = angle_delta

            if strategy == 1:
                pressure_ago = int(anchor_ago + diff[i])  # red line --> agonist
                pressure_ant = int(anchor_ant - diff[i])  # green line --> antagonist
            elif strategy == 2:
                if diff[i] > 0:
                    pressure_ago = int(abs(diff[i])) + anchor_ago
                    pressure_ant = anchor_ant
                else:
                    pressure_ago = anchor_ago
                    pressure_ant = int(abs(diff[i])) + anchor_ant

            frontend.add_command(dof, pressure_ago, pressure_ant,
                                o80.Iteration(iteration),
                                o80.Mode.QUEUE)
            frontend.add_command(dof, pressure_ago, pressure_ant,
                                o80.Iteration(iteration + iterations_per_command - 1),
                                o80.Mode.QUEUE)
            frontend.pulse()

            iteration += iterations_per_command
            theta_list = frontend.latest().get_positions()
            theta      = theta_list[dof]

        iteration_end  = iteration
        iteration = iteration_begin

        while iteration < iteration_end:
            observation = frontend.read(iteration)

            obs_pressure = observation.get_observed_pressures()
            des_pressure = observation.get_desired_pressures()
            obs_position = observation.get_positions()
            obs_velocity = observation.get_velocities()

            obs_pressure_ago = np.append(obs_pressure_ago, obs_pressure[dof][0])
            obs_pressure_ant = np.append(obs_pressure_ant, obs_pressure[dof][1])

            des_pressure_ago = np.append(des_pressure_ago, des_pressure[dof][0])
            des_pressure_ant = np.append(des_pressure_ant, des_pressure[dof][1])

            position = np.append(position, obs_position[dof])
            velocity = np.append(velocity, obs_velocity[dof])

            t_stamp = np.append(t_stamp, observation.get_time_stamp() * 1e-9)

            iteration += iterations_per_command

        print("...completed")
        
        initial_time = t_stamp[0]

        for i in range(0, len(t_stamp)):
            t_stamp[i] = t_stamp[i] - initial_time

        print("total duration:",t_stamp[-1]-t_stamp[0])
        print("expected:",t_stamp_u[-1])
        print("number of simulation:",len(t_stamp))
        print("desired number:",len(t_stamp_u))

        print("begin to write data to the file...")
        f = open(f_output, 'w')
        f.write(str(strategy))
        f.write("\n")
        f.write(str(p))
        f.write("\n")
        f.write(str(m))
        f.write("\n")
        f.write(str(N))
        f.write("\n")
        np.savetxt(f, t_stamp,          fmt='%.5f')
        np.savetxt(f, u_temp,           fmt='%.8f')  # desired input
        np.savetxt(f, diff,             fmt='%.8f')  # desired input
        np.savetxt(f, des_pressure_ago, fmt='%.5f') 
        np.savetxt(f, des_pressure_ant, fmt='%.5f')
        np.savetxt(f, obs_pressure_ago, fmt='%.5f')
        np.savetxt(f, obs_pressure_ant, fmt='%.5f')
        np.savetxt(f, position,         fmt='%.8f')
        f.close()
        print('...completed')

