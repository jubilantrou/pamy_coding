'''
This script is used to train PAMY2 with ILC.
'''
# %% import libraries
import PAMY_CONFIG
import os
import numpy as np
import o80_pam
import pickle5 as pickle
import time
from OCO_utils import *
from RealRobotGeometry import RobotGeometry
import wandb

# %% set parameters and initialize the robot
number_iteration = 35
root             = os.getcwd()
root_data        = os.path.join(root, 'data', 'ilc_with_increased_speed')
mkdir(root_data)

frontend         = o80_pam.FrontEnd("real_robot")
Pamy             = PAMY_CONFIG.build_pamy(frontend=frontend)

Pamy.LQITesting(t_start = 0.0, t_duration = 5.0)
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())

# %% do the training with ILC
wandb.init(
    entity='jubilantrou',
    project='pamy_oco_trial',
)

fix_seed(3154)
index_list = []
[index_list.append(count) for count in range(60)]

for index in index_list:
    '''
    1. update the desired trajectories
    2. update the optimizer for each dof
    3. train the robot with ILC algorithm
    '''
    start = time.time()

    ### the 1st step
    (t, angle) = get_random()
    RG = RobotGeometry(initial_posture=PAMY_CONFIG.GLOBAL_INITIAL)
    # (p, v, a, j, theta, t_stamp, vel_int) = RG.PathPlanning(time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, part=0, target_vel=4)
    (p, v, a, j, theta, t_stamp, vel_int, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
            time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method='no_delay', target_vel=4)
    theta = np.vstack((theta, np.zeros((1, theta.shape[1]))))
    theta_ = theta # absolute values
    theta = theta - theta[:, 0].reshape(-1, 1) # relative values    
    Pamy.ImportTrajectory(theta, t_stamp)

    ### the 2nd and the 3rd step
    Pamy.GetOptimizer(angle_initial_read, total_iteration=number_iteration, mode_name='none')

    (y_history, repeated, ff_history, disturbance_history, \
    P_history, d_lifted_history, P_lifted_history, \
    fb_history, ago_history, ant_history, y_pid) = Pamy.ILC(number_iteration=number_iteration, 
                                                            GLOBAL_INITIAL=PAMY_CONFIG.GLOBAL_INITIAL,
                                                            mode_name='none', ref_traj=theta_, T_go=T_go_list[-1])

    ### save useful results
    T = t_stamp[-1] - 1.5
    T_back = 1.35
    T_steady = 0.15
    t_list = np.array([0, T, T+T_back, T+T_back+T_steady])

    root_file = root_data + '/' + str(index)
    file = open(root_file, 'wb')
    # pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    # pickle.dump(angle_initial_read, file, -1)
    # pickle.dump(y_history, file, -1)
    # pickle.dump(repeated, file, -1)
    # pickle.dump(y_pid, file, -1)
    # pickle.dump(ff_history, file, -1)
    # pickle.dump(fb_history, file, -1)
    # pickle.dump(ago_history, file, -1)
    # pickle.dump(ant_history, file, -1)
    # pickle.dump(disturbance_history, file, -1)
    # pickle.dump(P_history, file, -1)
    # pickle.dump(d_lifted_history, file, -1)
    # pickle.dump(P_lifted_history, file, -1)
    pickle.dump(Pamy.y_desired.shape[1], file, -1)
    pickle.dump(Pamy.y_desired, file, -1)
    pickle.dump(disturbance_history[-1], file, -1)
    pickle.dump(ff_history[-1], file, -1)
    pickle.dump(t_list, file, -1)
    file.close()

    angle_initial_read =np.array(frontend.latest().get_positions())

    end = time.time()
    print('consumed time for index {}: {}s'.format(index, (end-start)))
