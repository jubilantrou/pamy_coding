'''
This script is used to do the system identification
'''
import numpy as np
import o80
import os
from get_handle import get_handle
import o80_pam
import PAMY_CONFIG
# %% constant
anchor_ago_list        = PAMY_CONFIG.anchor_ago_list
anchor_ant_list        = PAMY_CONFIG.anchor_ant_list
dof                    = 3 # which dof to excite
fs                     = 100
frequency              = 500.0  # frequency of the backend 
period                 = 1.0/frequency  # period of backend
duration_per_command   = 1 / fs  # period of frontend
iterations_per_command = int(duration_per_command/period)  # sychronize the frontend and the backend
anchor_ago             = anchor_ago_list[dof]
anchor_ant             = anchor_ant_list[dof]
amp_list               = [PAMY_CONFIG.pressure_limit[dof]/1000]
print('dof {}: ago is {}, ant is {}, and amp is {}'.format(dof, anchor_ago, anchor_ant, amp_list))
# %% connect to the simulation and initilize the posture
# handle           = get_handle(mujoco_id='SI',mode='pressure',generation='second')
# frontend         = handle.frontends["robot"]
segment_id = "real_robot"
frontend = o80_pam.FrontEnd(segment_id)
duration         = o80.Duration_us.seconds(2)
frontend.add_command(anchor_ago_list, anchor_ant_list,
                     duration,
                     o80.Mode.QUEUE)
frontend.pulse_and_wait()
# %%
p         = 10  # how many periods
file_path = '/home/mtian/Pamy_OCO/'
f_input   = file_path + 'excitation signals/excitation_5Hz.csv'
u         = np.loadtxt(f_input, delimiter=',')
[m, N]    = u.shape
u         = np.tile(u, p).flatten()
t_stamp_u = np.arange(0, (m * p * N)/fs, 1/fs)
t         = 1 / fs
T         = m * p * N / fs
# %% do the identification
for amp in amp_list:

    frontend.add_command(anchor_ago_list, anchor_ant_list,
                     duration,
                     o80.Mode.QUEUE)
    frontend.pulse_and_wait()

    u_temp = u * amp  #change the amplitude of the signal
    f_output = file_path + "5Hz_response_pamy2_real_dof_" + str(dof) + '_amp_' + str(amp) + ".txt"
    # initilization
    position = np.array([])
    velocity = np.array([])
    obs_pressure_ant = np.array([])
    obs_pressure_ago = np.array([])
    des_pressure_ant = np.array([])
    des_pressure_ago = np.array([])
    t_stamp = np.array([])

    ref_iteration   = frontend.latest().get_iteration()  # read the current iteration
    iteration_begin = ref_iteration + 1000  # set the iteration when beginning
    iteration       = iteration_begin
    
    print("begin to excite the mujoco...")
    for i in range(0, len(u_temp)):
        diff = int( u_temp[i] )
        
        # hardware paper as ref
        pressure_ago = int( anchor_ago + diff )  # red line --> agonist
        pressure_ant = int( anchor_ant - diff ) # green line --> antagonist

        frontend.add_command(dof, pressure_ago, pressure_ant,
                            o80.Iteration(iteration),
                            o80.Mode.QUEUE)

        frontend.add_command(dof, pressure_ago, pressure_ant,
                            o80.Iteration(iteration + iterations_per_command - 1),
                            o80.Mode.QUEUE)

        frontend.pulse()
        iteration += iterations_per_command

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
    f.write(str(p))
    f.write("\n")
    f.write(str(m))
    f.write("\n")
    f.write(str(N))
    f.write("\n")
    np.savetxt(f, t_stamp,          fmt='%.5f')
    np.savetxt(f, u_temp,           fmt='%.8f')
    np.savetxt(f, des_pressure_ago, fmt='%.5f')
    np.savetxt(f, des_pressure_ant, fmt='%.5f')
    np.savetxt(f, obs_pressure_ago, fmt='%.5f')
    np.savetxt(f, obs_pressure_ant, fmt='%.5f')
    np.savetxt(f, position,         fmt='%.8f')
    f.close()
    print('...completed')