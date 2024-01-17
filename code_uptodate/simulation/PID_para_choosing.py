'''
This script is used to find proper groups of PID parameters for the robot.
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
import time
from RealRobotGeometry import RobotGeometry
import time
import o80
import wandb
# %% parameter configuration file (can be organized as an independent file later)
obj = 'sim'
# parameters about the simulator setting in get_handle.py
# %% initialize the robot
if obj=='sim':
    handle   = get_handle()
    frontend = handle.frontends["robot"]
elif obj=='real':
    frontend = o80_pam.FrontEnd("real_robot")
else:
    raise ValueError('The variable obj needs to be assigned either as sim or as real!')

Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
RG = RobotGeometry()
# %%
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
# %%
choice = 0
amp = 10/180*math.pi

angle = np.array([0.0,0.0,0.0,0.0])
angle[choice] = amp
print(angle)

(step, t, position) = Pamy.AngleInitialization(angle = angle, choice = choice)

if_plot = 1

if if_plot:
    legend_position = 'best'
    fig = plt.figure(figsize=(18, 18))

    ax_position0 = fig.add_subplot(311)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Position of Dof_0 in degree')
    line = []
    line_temp, = ax_position0.plot(t, position[choice, :] * 180 / math.pi, linewidth=2)
    line.append( line_temp )
    line_temp, = ax_position0.plot(t, step * 180 / math.pi, linewidth=2)
    line.append( line_temp )
    # plt.legend(handles=line, loc=legend_position, shadow=True)

    plt.grid()
    plt.suptitle('Joint Space Output of Dof '+str(choice))              
    plt.show()