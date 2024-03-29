'''
This script is used to train the robot with ILC
'''
import PAMY_CONFIG
import math
import os
import numpy as np
import o80
import o80_pam
import matplotlib.pyplot as plt
import pickle5 as pickle
import time
import utils as fcs
from OCO_funcs import get_random, fix_seed
from RealRobotGeometry import RobotGeometry
import wandb
# %% set parameters
number_iteration = 30
root             = os.getcwd()
root_data        = os.path.join(root, 'data', 'ilc')
fcs.mkdir(root_data)
# %%
frontend         = o80_pam.FrontEnd("real_robot")  # conncet to the real robot
Pamy             = PAMY_CONFIG.build_pamy(frontend=frontend)  # build the physical model of pamy
# %%
# Pamy.AngleInitialization(PAMY_CONFIG.GLOBAL_INITIAL)  # set the initial posture for ILC
(ig_t, ig_step, ig_position, ig_diff, ig_theta_zero) = Pamy.LQRTesting(amp = np.array([[30], [30], [30]])/180*math.pi, t_start = 0.0, t_duration = 5.0)
Pamy.PressureInitialization()  # set the initial pressure for ILC
angle_initial_read = np.array(frontend.latest().get_positions())  # read the current posture of Pamy from the sensors

# %% train the ILC
fix_seed(3154)

wandb.init(
    entity='jubilantrou',
    project='pamy_oco_trial',
)


index_list = [0]
for index in index_list:
    start = time.time()
    '''
    1. update the desired trajectories
    2. update the optimizer for each dof
    3. train the robot with ILC algorithm
    '''
    (t, angle) = get_random()
    RG = RobotGeometry(initial_posture=PAMY_CONFIG.GLOBAL_INITIAL)
    (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=PAMY_CONFIG.GLOBAL_INITIAL, T_go=t, target=angle, part=0)
    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = theta
    theta = theta - theta[:, 0].reshape(-1, 1)
    p = theta

    T = t_stamp[-1] - 1.7
    T_back = 1.5
    T_steady = 0.2
    Pamy.ImportTrajectory(p, t_stamp)
    Pamy.GetOptimizer(angle_initial_read, total_iteration=number_iteration, mode_name='none')
    (y_history, repeated, ff_history, disturbance_history, \
    P_history, d_lifted_history, P_lifted_history, \
    fb_history, ago_history, ant_history, y_pid) = Pamy.ILC(number_iteration=number_iteration, 
                                                            GLOBAL_INITIAL=PAMY_CONFIG.GLOBAL_INITIAL,
                                                            mode_name='none',ref_traj=theta_)

    t_list = np.array([0, T, T+T_back, T+T_back+T_steady])

    root_file = root_data + '/' + str(index)
    file = open(root_file, 'wb')
    pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    pickle.dump(t_list, file, -1)
    pickle.dump(angle_initial_read, file, -1)
    pickle.dump(y_history, file, -1)
    pickle.dump(repeated, file, -1)
    pickle.dump(y_pid, file, -1)
    pickle.dump(ff_history, file, -1)
    pickle.dump(fb_history, file, -1)
    pickle.dump(ago_history, file, -1)
    pickle.dump(ant_history, file, -1)
    pickle.dump(disturbance_history, file, -1)
    pickle.dump(P_history, file, -1)
    pickle.dump(d_lifted_history, file, -1)
    pickle.dump(P_lifted_history, file, -1)
    file.close()

    angle_initial_read =np.array(frontend.latest().get_positions())
    end = time.time()
    print(end - start)