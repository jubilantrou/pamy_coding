'''
This script is used to decide the ultimate gain and the oscillation period, 
which can be used to find proper groups of PID parameters for the robot 
with Ziegler-Nichols method.
'''
# %% import libraries
import PAMY_CONFIG
import math
import os
import matplotlib.pyplot as plt
from get_handle import get_handle
import o80_pam
from RealRobotGeometry import RobotGeometry

# %% set parameters
obj = 'sim' # for the simulator or the real robot
choice = 2 # which Dof to do test for
amp = 5/180*math.pi # the increased amplitude of the input step signal based on the initial position
t_start = 0.5 # the starting time of the step signal
t_duration = 5 # the whole time length for recording and plotting

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

def plot(t, ref, result, choice):
    fig = plt.figure(figsize=(18, 18))
    ax_position0 = fig.add_subplot(111)
    plt.xlabel(r'Time in s')
    plt.ylabel(r'Position of Dof_' + str(choice) + 'in degree')
    line = []
    line_temp, = ax_position0.plot(t, ref * 180 / math.pi, linewidth=2, label='reference signal')
    line.append( line_temp )
    line_temp, = ax_position0.plot(t, result * 180 / math.pi, linewidth=2, label='output signal')
    line.append( line_temp )
    plt.legend(handles=line, shadow=True)
    plt.grid()
    plt.suptitle('Joint Space for Dof '+str(choice))              
    plt.show()

# %% run the main
if __name__ == '__main__':
    (t, step, position) = Pamy.PIDTesting(choice = choice, amp = amp, t_start = t_start, t_duration = t_duration)
    plot(t, step, position, choice)
    # TODO: do the following procedure via scripts
