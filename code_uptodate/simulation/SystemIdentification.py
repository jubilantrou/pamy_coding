'''
This script is used to do the system identification by exciting the DoFs seperately
using the random multisine signal.
'''
import numpy as np
import o80
from get_handle import get_handle
import o80_pam
import PAMY_CONFIG

# %% set constants that will be used
anchor_ago_list        = PAMY_CONFIG.anchor_ago_list
anchor_ant_list        = PAMY_CONFIG.anchor_ant_list
dof                    = 2 # which DoF to excite
anchor_ago             = anchor_ago_list[dof]
anchor_ant             = anchor_ant_list[dof]
frequency              = 500.0 # frequency of the backend 
period                 = 1.0/frequency # period of the backend
fs                     = 100 # frequency of the frontend
duration_per_command   = 1 / fs # period of the frontend
iterations_per_command = int(duration_per_command/period) # to sychronize the frontend and the backend
amp_list               = [PAMY_CONFIG.pressure_limit[dof]/1000] # the maximum allowed amp
amp_list               = [3.0]
print('dof {}: anchor ago pressure is {}, anchor ant pressure is {}, and amp is {}'.format(dof, anchor_ago, anchor_ant, amp_list))

# %% connect to the real robot or to the simulator and initilize it
# connect to the real robot
frontend = o80_pam.FrontEnd("real_robot")
# connect to the simulator
# handle = get_handle()
# frontend = handle.frontends["robot"]
Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)

# %% set parameters for the SI
p         = 6  # how many periods used for repeat
file_path = '/home/mtian/Pamy_OCO/'
f_input   = file_path + 'excitation signals/excitation_3Hz.csv' # the file storing the excitation signal
u         = np.loadtxt(f_input, delimiter=',')
u         = u[0:5,:] # each row in u corresponds to a signal, and here we pick out the first 5 signals
[m, N]    = u.shape # N is number of time points in each signal
u         = np.tile(u, p).flatten() # we will repeat the picked m signals by p times
t_stamp_u = np.arange(0, (m * p * N)/fs, 1/fs)
t         = 1 / fs
T         = m * p * N / fs
print('estimated time: {}s'.format(T))

# %% do the SI
# TODO: need to ensure that the robot is already at its desired initial setting before starting the SI, either by 
# adjusting it manually after pressure initialization or by using the PID/LQR controller that you have tuned
for amp in amp_list:
    u_temp = u * amp  # increase the amplitude of the signal
    f_output = file_path + "new_3Hz_response_pamy2_real_dof_" + str(dof) + '_amp_' + str(amp) + ".txt" # the file to store the response
    
    position = np.array([])
    velocity = np.array([])
    obs_pressure_ant = np.array([])
    obs_pressure_ago = np.array([])
    des_pressure_ant = np.array([])
    des_pressure_ago = np.array([])
    t_stamp = np.array([])

    ref_iteration   = frontend.latest().get_iteration()  # read the current iteration
    iteration_begin = ref_iteration + 500  # set the starting iteration
    iteration       = iteration_begin
    
    frontend.add_command(anchor_ago_list, anchor_ant_list,
                        o80.Iteration(iteration_begin-1),
                        o80.Mode.QUEUE)
    frontend.pulse()

    print("begin to excite the mujoco...")
    for i in range(0, len(u_temp)):
        diff = int(u_temp[i])
        
        pressure_ago = int( anchor_ago + diff )
        pressure_ant = int( anchor_ant - diff )

        frontend.add_command(dof, pressure_ago, pressure_ant,
                            o80.Iteration(iteration),
                            o80.Mode.QUEUE)
        frontend.pulse()
        frontend.add_command(dof, pressure_ago, pressure_ant,
                            o80.Iteration(iteration + iterations_per_command - 1),
                            o80.Mode.QUEUE)
        frontend.pulse_and_wait()

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
        position         = np.append(position, obs_position[dof])
        velocity         = np.append(velocity, obs_velocity[dof])
        t_stamp          = np.append(t_stamp, observation.get_time_stamp() * 1e-9)

        iteration += iterations_per_command
    print("...completed")
    
    initial_time = t_stamp[0]
    for i in range(0, len(t_stamp)):
        t_stamp[i] = t_stamp[i] - initial_time

    print("total duration:", t_stamp[-1]-t_stamp[0])
    print("expected duration:", t_stamp_u[-1])
    print("number of collected points:", len(t_stamp))
    print("desired number of collected points:", len(t_stamp_u))

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
