'''
This script is used to test the performance of the LQR/LQI controller got from the MATLAB script 
based on the MIMO system identification result using the least square method.
'''
# %% import libraries
import PAMY_CONFIG
import math
import matplotlib.pyplot as plt
from get_handle import get_handle
import o80_pam
import numpy as np
from OCO_utils import *
from RealRobotGeometry import RobotGeometry

# %% create functions
def plot(t, ref, result, u):
    fig = plt.figure(figsize=(18, 18))

    ax_position0 = fig.add_subplot(311)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_1 in degree')
    line = []
    line_temp, = ax_position0.plot(t, ref[0::3] * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position0.plot(t, result[0::3] * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )
    plt.legend(handles=line, shadow=True)

    ax_position1 = fig.add_subplot(312)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_2 in degree')
    line = []
    line_temp, = ax_position1.plot(t, ref[1::3] * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position1.plot(t, result[1::3] * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )
    plt.legend(handles=line, shadow=True)

    ax_position2 = fig.add_subplot(313)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_3 in degree')
    line = []
    line_temp, = ax_position2.plot(t, ref[2::3] * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position2.plot(t, result[2::3] * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )
    plt.legend(handles=line, shadow=True)

    fig1 = plt.figure(figsize=(18, 18))

    ax_position3 = fig1.add_subplot(111)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Pressures')
    line = []
    line_temp, = ax_position3.plot(t, u[0::3], linewidth=2, label='Dof_1')
    line.append( line_temp )
    line_temp, = ax_position3.plot(t, u[1::3], linewidth=2, label='Dof_2')
    line.append( line_temp )
    line_temp, = ax_position3.plot(t, u[2::3], linewidth=2, label='Dof_3')
    line.append( line_temp )

    plt.legend(handles=line, shadow=True)
    plt.grid()
    plt.suptitle('Joint Space Movement')              
    plt.show()

# %% run the main
if __name__ == '__main__':
    obj = 'real'
    amp = np.array([[-20], [-20], [-20]])/180*math.pi # the offsets from the starting position that are desired to be tracked
    t_start = 0.0 # the starting time of using controller
    t_duration = 5.0 # the whole time length for control performance test

    if obj=='sim':
        handle   = get_handle()
        frontend = handle.frontends["robot"]
    elif obj=='real':
        frontend = o80_pam.FrontEnd("real_robot")
    else:
        raise ValueError('The variable obj needs to be assigned either as sim or as real!')

    if obj != PAMY_CONFIG.obj:
        raise ValueError("Make sure the value of obj in PAMY_CONFIG is the same as the one we specify above!")

    Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
    RG = RobotGeometry(initial_posture=PAMY_CONFIG.GLOBAL_INITIAL)

    option = 1 # choose option from {1, 2, 3}

    # option 1: test the perfomance of angle initialization using the LQI controller
    if option==1:
        Pamy.PressureInitialization()

        Pamy.LQITesting(t_start = 0.0, t_duration = 6.0)
        print('completed angle initialization:')
        print(np.array(frontend.latest().get_positions())/math.pi*180)

        Pamy.PressureInitialization(duration=4)
        print('completed pressure initialization:')
        print(np.array(frontend.latest().get_positions())/math.pi*180)
        print(frontend.latest().get_observed_pressures())

    # option 2: test the performance of set point tracking using the LQI controller and the performance of regulating using the LQR controller
    elif option==2:
        Pamy.PressureInitialization()

        (t, pos_ref, position, u_LQI, theta_begin) = Pamy.LQITesting(t_start = t_start, t_duration = t_duration, amp = amp)
        plot(t, pos_ref, position, u_LQI)

        # TODO: need to tune parameters for LQRTestingFollowup() again after the repairing of PAMY
        (t_followup, pos_ref_followup, position_followup, u_LQR) = Pamy.LQRTestingFollowup(tar = PAMY_CONFIG.GLOBAL_INITIAL, t_duration = t_duration)
        plot(t_followup, pos_ref_followup, position_followup, u_LQR)

        Pamy.PressureInitialization(duration=2)

    # option 3: test the performance of reference tracking using the LQI controller
    elif option==3:
        Pamy.LQITesting(t_start = 0.0, t_duration = 6.0)
        Pamy.PressureInitialization()

        fix_seed(3131)
        (t, angle) = get_random()
        (p, v, a, j, theta, t_stamp, vel_int) = RG.PathPlanning(time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, part=0)
        theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
        theta_ = np.copy(theta) # absolute values of the reference
        theta = theta - theta[:, 0].reshape(-1, 1) # relative values of the reference

        # TODO: need to tune parameters for LQITrackingTesting() again after the repairing of PAMY
        (t, pos_ref, position, u_LQI, theta_begin) = Pamy.LQITrackingTesting(ref_traj = theta)
        plot(t, pos_ref, position, u_LQI)
        
        Pamy.PressureInitialization(duration=2)
