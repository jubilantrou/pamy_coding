import time
import math
import os
import numpy as np
import o80
import o80_pam
from RealRobot import Robot
import pickle5 as pickle
import RealTrajectoryGeneration
from RealJoint import Joint as RealJoint
import matplotlib.pyplot as plt
# %%
def FulfillZeros( a ):
    b = np.zeros(  [len(a), len(max(a, key = lambda x: len(x)))]  )
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b
# %%
# connect to the simulation
frontend = o80_pam.FrontEnd("real_robot")
# %% constant variables
# sampling frequency of the desired trajectory 
fs = 100
# frequency of the backend
frequency = 500.0
# the list for all dofs
dof_list = [0, 1, 2, 3]
# %% some model parameters
# situation 1
delta_d_list = np.array([1e-9, 1e-11, 1e-8, 1e-9]) 
delta_y_list = np.array([1e-3, 1e-3, 1e-3, 1e-3])
delta_w_list = np.array([1e-9, 1e-11, 1e-8, 1e-9])
delta_ini_list = np.array([1e-7, 1e-7, 1e-7, 1e-7])
# import the linear inverse model
inverse_model_num = [[-1.603053083740808, 4.247255623123213, -3.894718257376192, 1.248397916673204],
                     [39.4562899165285, -107.6338556336952, 101.4865793303678, -33.2833280994107],
                     [-0.5340642923467604, 0.9405453623185243, -0.4103846610378554],
                     [1.139061120808877, -1.134002042583525]]
inverse_model_num = FulfillZeros( inverse_model_num ) * 1e5

inverse_model_den = [[1.000000000000000, -2.397342550890251, 2.136262918863130, -0.683005338583667],
                     [1.000000000000000, -1.972088694237271, 1.604586816873790, -0.428252150060600],
                     [1.000000000000000, -1.702657081211864, 0.823186864989291],
                     [1.000000000000000, -0.825587854345462]]
inverse_model_den = FulfillZeros( inverse_model_den )
# the delay of the inverse model
ndelay_list = np.array([2, 2, 3, 1])
# the order of the inverse model
order_num_list = np.array([3, 3, 2, 1])
order_den_list = np.array([3, 3, 2, 1])
# import the linear model
model_num = [[-0.62380966054252, 1.49548544287500, -1.33262144624559, 0.42606532841061],
             [0.0253445015260062, -0.0499816049205160, 0.0406674530288672, -0.0108538372707263],
             [-1.87243373940214, 3.18811256549306, -1.54136285983862],
             [0.877916014980719, -0.724796799103451]]
model_num = FulfillZeros( model_num ) * 1e-5

model_den = [[1.000000000000000, -2.649479088497819, 2.429562874042614, -0.778762680621906],
             [1.000000000000000, -2.727926418358119, 2.572126764707656, -0.843549359806079],
             [1.000000000000000, -1.761108869843411, 0.768418085460389],
             [1.000000000000000, -0.995558554204924]]
model_den = FulfillZeros( model_den )   

model_num_order = [3, 3, 2, 1]
model_den_order = [3, 3, 2, 1]
model_ndelay_list = [2, 2, 3, 1]
# the set point pressure for each dof
anchor_ago_list = np.array([17500, 18500, 16000, 15000])
anchor_ant_list = np.array([17500, 18500, 16000, 15000])
# the strategy code for each dof
strategy_list = np.array([2, 2, 2, 2])
# parameters of PID controller for each dof
pid_list = [[-3.505924158687806e+04, -3.484022215671791e+05 / 5, -5.665386729745434e+02],
            [8.228984656729296e+04, 1.304087541343074e+04, 4.841489121599795e+02],
            [-36752.24956301624, -246064.5612272051/ 10, -531.2866756516057],
            [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]]
pid_list = FulfillZeros( pid_list )
# dof_0: -20000, -794351 / 6.0, -2547 / 2.0 aggressive design
# dof_0: -2.005924158687806e+04, -3.484022215671791e+05 / 1.5, -5.665386729745434e+02 good result
# dof_0: -2.505924158687806e+04, -3.484022215671791e+05 / 2, -5.665386729745434e+02 best one
# dof_1: 8.228984656729296e+04 * 0.8, 1.304087541343074e+04 * 12.0, 4.841489121599795e+02
#dof_2: -36752.24956301624 * 0.5, -246064.5612272051 / 3, -531.2866756516057

controller_name = 'pid_manually'

if controller_name == 'pid_grid_search':
    # old one (form grid search)
    pid = np.array([[-13000, 0, -300],
                    [80000, 0, 300],
                    [-5000, -8000, -100],
                    [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])
if controller_name == 'pid_manually':    
    # new one (manually tunning)
    pid = np.array([[-2.505924158687806e+04, -3.484022215671791e+05 / 2, -5.665386729745434e+02],
                    [8.228984656729296e+04 * 0.4, 1.304087541343074e+04 * 8.0, 4.841489121599795e+02],
                    [-36752.24956301624 * 0.100, -246064.5612272051/ 100, -531.2866756516057 / 4],
                    [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])
if controller_name == 'pid_tc_2':
    # tc = 2
    pid = np.array([[-5000, -6000, 0],
                    [31000, 2000, 200],
                    [-2000, -5000, 0],
                    [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])
if controller_name == 'pid_tc_6':    
    # tc = 6
    pid = np.array([[-14000, -15000, -300],
                    [64000, 30000, 1900],
                    [-4000, -13000, -100],
                    [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])
if controller_name == 'pid_tc_10':    
    # tc = 10
    pid = np.array([[-31000, -16000, -300],
                    [88000, 40000, 1700],
                    [-6000, -18000, -100],
                    [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])

# constraints for each muscle
pressure_min = [-4500, -6500, -6000, -7000]
pressure_max = [4500, 6500, 6000, 7000]
# %%
angle_initial = frontend.latest().get_positions()
angle_initial = np.array( angle_initial )
# %% generate step signal
path_to_save = "/home/hao/Desktop/Hao/prediction_result/PIDComparison"

T = 3
Ns = 301
amplitude_u = 10 / (1*180/math.pi)

step_signal = np.ones( Ns ) * amplitude_u
step_signal[0:101] = 0

t = 1 / fs
t_stamp_u = np.linspace(0, T, Ns)
zero_signal = np.zeros( len(t_stamp_u) )

for dof in range( 3 ):
    

    y_desired = np.array([])
    for idx in range(4):
        if idx == dof:
            y_desired = np.append(y_desired, step_signal) 
        else:
            y_desired = np.append(y_desired, zero_signal)  
    y_desired = y_desired.reshape(4, -1)

    # %% define the robot
    RealRobot = Robot(frontend, dof_list, y_desired, t_stamp_u, model_num, model_den,
                    model_num_order, model_den_order, model_ndelay_list,
                    inverse_model_num, inverse_model_den, order_num_list, order_den_list,
                    ndelay_list, anchor_ago_list, anchor_ant_list, strategy_list, pid_list,
                    delta_d_list, delta_y_list, delta_w_list, delta_ini_list,
                    pressure_min, pressure_max)

    (y_together, _) = RealRobot.Control(y_list=y_desired, 
                                        mode_name_list=["fb", "fb", "fb", "fb"], 
                                        mode_trajectory="ref",
                                        ifplot="no", echo="yes",
                                        controller=controller_name)
    RealRobot.AngleInitialization( angle_initial, mode_name='Together')
    RealRobot.PressureInitialization()
    # %% define the second joint
    Joint = RealJoint(frontend, dof, anchor_ago_list[dof], anchor_ant_list[dof], 
                  inverse_model_num[dof], inverse_model_den[dof],
                  order_num_list[dof], order_den_list[dof], ndelay_list[dof],
                  pid[dof], strategy_list[dof])
    # %%
    (y_single, _, _) = Joint.Control(y_desired[dof, :], mode_name='fb', ifplot='no', echo='yes')

    RealRobot.AngleInitialization( angle_initial, mode_name='Together')
    RealRobot.PressureInitialization()    
    # %% plot
    plt.figure( num=dof, figsize=(16, 8))
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Angle $\theta$ in degree')
    plt.plot(t_stamp_u, y_desired[dof, :] * 180 / math.pi, 
            linestyle='--', linewidth=1.5, label=r'Step' )
    plt.plot(t_stamp_u, (y_together[dof, :] - y_together[dof, 0])* 180 / math.pi, 
            color='royalblue', linewidth=2, label=r'Together' )
    plt.plot(t_stamp_u, (y_single - y_single[0])* 180 / math.pi, color='lightgreen',
            linewidth=2, label=r'Single')
    plt.legend(ncol=3, loc='lower right', shadow=True)
    # plt.show()        
    plt.savefig(path_to_save + '/' + 'dof_' + str(dof) + '/' + controller_name + '.pdf')
    plt.close()