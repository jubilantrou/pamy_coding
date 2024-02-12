'''
This script is used to decide the ultimate gain and the oscillation period 
of each Dof, which can be used to find proper groups of PID parameters for 
the robot with the Ziegler-Nichols method.
'''
# %% import libraries
import PAMY_CONFIG
import math
import os
import matplotlib.pyplot as plt
from get_handle import get_handle
import o80_pam
from RealRobotGeometry import RobotGeometry
import scipy

# %% set parameters
obj = 'sim' # for the simulator or the real robot
choice = 2 # for which Dof to do the experiment
amp = 25/180*math.pi # the increased amplitude of the input step signal based on the initial position
t_start = 0.5 # the starting time of the step signal
t_duration = 5.0 # the whole time length for recording and plotting
mode1 = 'no overshoot' # the name of control type for Ziegler-Nichols method
mode2 = 'classic PID'
mode3 = 'Pessen Integral Rule'

# %% initialize the chosen obj
if obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
else:
    raise ValueError('The variable obj needs to be assigned either as sim or as real!')

Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
RG = RobotGeometry()

# %% create functions
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def plot(t, ref, result, choice, p_ago=None, p_ant=None, peaks=None):
    fig = plt.figure(figsize=(18, 18))
    ax_position0 = fig.add_subplot(121)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_' + str(choice) + 'in degree')
    line = []
    line_temp, = ax_position0.plot(t, ref * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position0.plot(t, result * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )
    if peaks is not None:
        line_temp, = ax_position0.plot(t[peaks], result[peaks] * 180 / math.pi, 'x', label='output peaks detected')
        line.append( line_temp )
    
    # ax_position1 = fig.add_subplot(122)
    # plt.xlabel(r'Time in s')
    # plt.ylabel(r'Pressure of Dof_' + str(choice))
    # line = []
    # line_temp, = ax_position1.plot(t, p_ago, linewidth=2, label='ago pressure')
    # line.append( line_temp )
    # line_temp, = ax_position1.plot(t, p_ant, linewidth=2, label='ant pressure')
    # line.append( line_temp )

    plt.legend(handles=line, shadow=True)
    plt.grid()
    plt.suptitle('Joint Space for Dof '+str(choice))              
    plt.show()

def para_compute(Ku, Tu, mode):
    if (mode=='no overshoot'):
        Kp = 0.2*Ku
        Ki = 0.4*Ku/Tu
        Kd = 0.066*Ku*Tu
    elif (mode=='classic PID'):
        Kp = 0.6*Ku
        Ki = 1.2*Ku/Tu
        Kd = 0.075*Ku*Tu
    elif (mode=='Pessen Integral Rule'):
        Kp = 0.7*Ku
        Ki = 1.75*Ku/Tu
        Kd = 0.105*Ku*Tu
    return(Kp,Ki,Kd)

# %% run the main
if __name__ == '__main__':
    init = 0

    if init:
        Pamy.PressureInitialization(duration=4)
    
    else:
        (t, step, position) = Pamy.PIDTesting(choice = choice, amp = amp, t_start = t_start, t_duration = t_duration)
        Pamy.PressureInitialization(duration=4)

        ready_to_process = 0
        peaks = None
        if ready_to_process:
            peaks, _ = scipy.signal.find_peaks(position)
            num_peaks_taken = 3
            Tu = (t[peaks[-1]]-t[peaks[-1-num_peaks_taken]])/num_peaks_taken
            print('Tu: {}'.format(Tu))

            print('computation method: {}'.format(mode1))
            print('P, I, D: {}'.format(para_compute(Pamy.pid_list[choice,0],Tu,mode1)))

            print('computation method: {}'.format(mode2))
            print('P, I, D: {}'.format(para_compute(Pamy.pid_list[choice,0],Tu,mode2)))

            print('computation method: {}'.format(mode3))
            print('P, I, D: {}'.format(para_compute(Pamy.pid_list[choice,0],Tu,mode3)))

        plot(t, step, position, choice, peaks)
        # TODO: do the following procedure via scripts
