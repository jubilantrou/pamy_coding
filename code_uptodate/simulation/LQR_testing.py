# %% import libraries
import PAMY_CONFIG
import math
import os
import matplotlib.pyplot as plt
from get_handle import get_handle
import o80_pam
from RealRobotGeometry import RobotGeometry
import scipy
import numpy as np

# %% set parameters
obj = 'real'
amp = np.array([[-20], [-20], [-20]])/180*math.pi
t_start = 0.0
t_duration = 6.0

# %% initialize the chosen obj
if obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
else:
    raise ValueError('The variable obj needs to be assigned either as sim or as real!')

print(PAMY_CONFIG.obj)
Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
RG = RobotGeometry()

# %% create functions
def plot(t, ref, result, diff):
    fig = plt.figure(figsize=(18, 18))
    ax_position0 = fig.add_subplot(311)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_1 in degree')
    line = []
    line_temp, = ax_position0.plot(t, ref[0::3] * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position0.plot(t, result[0::3] * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )

    ax_position1 = fig.add_subplot(312)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_2 in degree')
    line = []
    line_temp, = ax_position1.plot(t, ref[1::3] * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position1.plot(t, result[1::3] * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )

    ax_position2 = fig.add_subplot(313)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_3 in degree')
    line = []
    line_temp, = ax_position2.plot(t, ref[2::3] * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position2.plot(t, result[2::3] * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )

    fig1 = plt.figure(figsize=(18, 18))
    ax_position3 = fig1.add_subplot(111)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Pressures')
    line = []
    line_temp, = ax_position3.plot(t, diff[0::3], linewidth=2, label='Dof_1')
    line.append( line_temp )
    line_temp, = ax_position3.plot(t, diff[1::3], linewidth=2, label='Dof_2')
    line.append( line_temp )
    line_temp, = ax_position3.plot(t, diff[2::3], linewidth=2, label='Dof_3')
    line.append( line_temp )

    plt.legend(handles=line, shadow=True)
    plt.grid()
    plt.suptitle('Joint Space Movement')              
    plt.show()

# %% run the main
if __name__ == '__main__':

    init = 1

    if init:
        # Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)
        Pamy.PressureInitialization(duration=4)
        print(np.array(frontend.latest().get_positions())/math.pi*180)
        print(frontend.latest().get_observed_pressures())
        print(PAMY_CONFIG.pressure_limit)

    else:
        # Pamy.PressureInitialization()
        (t, step, position, diff, theta_zero) = Pamy.LQRTesting(amp = amp, t_start = t_start, t_duration = t_duration)
        plot(t, step, position, diff)
        print(frontend.latest().get_observed_pressures())
        # Pamy.PressureInitialization(duration=2)

        # (t_fp, step_fp, position_fp, diff_fp) = Pamy.LQRTestingFollowup(tar = theta_zero, t_duration = t_duration)
        # plot(t_fp, step_fp, position_fp, diff_fp)
        # Pamy.PressureInitialization(duration=2)
