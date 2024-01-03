import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import o80
import o80_pam
from RealRobot import Robot
import RealRacket

path_of_file = "/home/hao/Desktop/Hao/day_15_08/" + '4' + '.txt'
print(path_of_file)
# path_of_file = "/home/hao/Desktop/Hao/data/" + "F_SmoothSignal_1_a__i_40.txt"
file = open(path_of_file, 'rb')
t_stamp_u = pickle.load(file) # time stamp for x-axis
y_history = pickle.load(file) 
ff_history = pickle.load(file) 
disturbance_history = pickle.load(file) 
P_history = pickle.load(file)
time_history = pickle.load(file) 
file.close()

y_desired = y_history[0]
ff = ff_history[-1]

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
# amplitude of desired trajectory
amplitude_input_list = np.array([2, -1, -2, 1])
# how many iterations
number_iteration = 100
# covariance parameter for each dof
# dof: 0 1 2 3
delta_d_list = np.array([1e-10, 1e-10, 1e-8, 1e-8]) 
delta_y_list = np.array([1e-2, 1e-2, 1e-3, 1e-3])
delta_w_list = np.array([1e-10, 1e-10, 1e-8, 1e-8])
delta_ini_list = np.array([1e-7, 1e-7, 1e-7, 1e-7])
# %%
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

pressure_list = [[],
                 [],
                 [21500, 21000, 20500, 20000, 19500, 19000, 18500, 18000, 17500, 17000, 16500, 16000, 15500, 15000, 14500],
                 []]
pressure_list = FulfillZeros( pressure_list )           

section_list = [[],
                [],
                [-1.0664, -1.0140, -0.9540, -0.8829, -0.7942, -0.6714, -0.4495, 0.0637, 0.4918, 0.6981, 0.8115, 0.8956, 0.9639, 1.0220, 1.0730],
                []]
section_list = FulfillZeros( section_list )
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

# pid_list = [[-13000, 0, -300],
#             [80000, 0, 300],
#             [-5000, -8000, -100],
#             [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]]
pid_list = FulfillZeros( pid_list )
# constraints for each muscle
pressure_min = [-4500, -6500, -6000, -7000]
pressure_max = [4500, 6500, 6000, 7000]
# %% generate the real robot
RealRobot = Robot(frontend, dof_list, y_desired, t_stamp_u, model_num, model_den,
                  model_num_order, model_den_order, model_ndelay_list,
                  inverse_model_num, inverse_model_den, order_num_list, order_den_list,
                  ndelay_list, anchor_ago_list, anchor_ant_list, strategy_list, pid_list,
                  delta_d_list, delta_y_list, delta_w_list, delta_ini_list,
                  pressure_min, pressure_max)
# %%
repeat_time = 1

angle_initial = frontend.latest().get_positions()
angle_initial = np.array( angle_initial )

( u_ago, u_ant, ff_temp ) = RealRobot.Feedforward( y_desired, angle_initial )

position_history = []
for i in range( repeat_time ):
    ( position, y_hat ) = RealRobot.Control( y_desired, mode_name_list=["ff+fb", "ff+fb", "ff+fb", "ff+fb"], 
                                             mode_trajectory="ref", u_ago=u_ago, u_ant=u_ant, 
                                             ff=ff, ifplot="no", echo="yes",
                                             controller='pid_tc_10')

    position_history.append( position )

    RealRobot.AngleInitialization( angle_initial, T=6 , mode_name='Together')
    RealRobot.PressureInitialization()
# %%
Location = "upper center"

line = []
plt.figure( figsize=(16, 8) )
plt.xlabel(r'Time $t$ in s')
plt.ylabel(r'Angle $\theta$ in degree')
for dof in dof_list:
    line_temp, = plt.plot(t_stamp_u, (y_desired[dof, :]) * 180 / math.pi,
                          linestyle='--', linewidth=2, 
                          label=r'Desired {}'.format(dof))
    line.append(line_temp)
    for i in range( repeat_time ):
        line_temp, = plt.plot(t_stamp_u, 
                              (position_history[i][dof, :]-position_history[i][dof, 0]) * 180 / math.pi, 
                              linestyle='-', linewidth=1,
                              label=r'Result {}'.format(dof))
        line.append(line_temp)

plt.legend(handles = line, loc=Location, shadow=True)
plt.show()

# for dof in dof_list:
#     line = []
#     plt.figure(dof)
#     plt.xlabel(r'Time $t$ in s')
#     plt.ylabel(r'Trajectory difference')
#     for i in range( repeat_time ):
#         line_temp, = plt.plot(t_stamp_u, ( position_history[i][dof, :]-position_history[0][dof, :] )* 180 / math.pi, label=r'Result {}'.format(dof), linewidth=0.3)
#         line.append(line_temp)

#     plt.legend(handles = line, loc=Location, shadow=True)
#     plt.show()