# %%
import time
import math
import os
import numpy as np
import o80
import o80_pam
from RealRobot import Robot
import pickle5 as pickle
from RealRobotGeometry import RobotGeometry as RG
import matplotlib.pyplot as plt
import torch
# %%
def FulfillZeros( a ):
    b = np.zeros(  [len(a), len(max(a, key = lambda x: len(x)))]  )
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b
# %% read data of balls
path_of_file = "/home/hao/Desktop/HitBall/" + 'BallsData' + '.txt'
# path_of_file = "/home/hao/Desktop/Hao/data/" + "F_SmoothSignal_1_a__i_40.txt"
file = open(path_of_file, 'rb')
time_list = pickle.load(file)
position_list = pickle.load(file)
velocity_list = pickle.load(file)
file.close()

RobotGeometry = RG()
# length of robot arms
l_1 = 0.4
l_2 = 0.38
# calculate mean position of all points
position_mean = np.mean( position_list, axis=0 )
# gathering all positions
# position_list = (position_list - position_mean) * 0.8 + position_mean
# offset angles of upright posture
offset = np.array( [2.94397627, -0.078539855235, -0.06333859293225] )
# angles of initial posture
angle_initial_ref = np.array( [2.94397627, -0.605516948, -0.5890489142699] )
# anchor angles (= mean position)
angle_anchor_ref = np.array( [ 2.94397627, -1.452987321865, -0.87660612618] )

angle_initial = angle_initial_ref - offset
angle_anchor = angle_anchor_ref - offset
# position_anchor is in cartesian space
# angle_anchor is in joint space
(_, position_anchor) = RobotGeometry.AngleToEnd(angle_anchor, frame='Cartesian')
position_error = position_anchor - position_mean
# list of positions, where to hit the balls
position_list = position_list + position_error  # have been calibrated
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

# delta_d_list = np.array([1e-9, 1e-11, 1e-11, 1e-9]) 
# delta_y_list = np.array([1e-3, 1e-3, 1e-3, 1e-3])
# delta_w_list = np.array([1e-9, 1e-11, 1e-11, 1e-9])
# delta_ini_list = np.array([1e-7, 1e-7, 1e-7, 1e-7])

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

pid_list = FulfillZeros( pid_list )
# constraints for each muscle
pressure_min = [-4500, -6500, -6000, -7000]
pressure_max = [4500, 6500, 6000, 7000]
# read the initial angles of robot arm
angle_initial_read = frontend.latest().get_positions()
angle_initial_read = np.array( angle_initial_read )
# %% online optimization for all dofs
path_root = "/home/hao/Desktop/HitBall/test" 

path_of_file = "/home/hao/Desktop/HitBall/cnn_model_2"

path_of_LModel = path_of_file + '/' + 'A_list'
file = open(path_of_LModel, 'rb')
A_list = pickle.load(file)
file.close()

A_np_list = []
for dof in range(3):
    A_np_list.append( A_list[dof].detach().numpy() )
A_np_list = np.array( A_np_list ).T
A_list = A_np_list[0:603, :]
# A_list = A_list.reshape(201, 3, 3)
A_bias = A_np_list[-1, :]
cnn_model_list = []

map_location = torch.device('cpu')

device = 'cpu'

for dof in range( 3 ):
    path_of_CModel = path_of_file + '/' + 'model_dof_' + str(dof)
    path_of_CModel_model = path_of_CModel + '/' + 'model'
    path_of_CModel_parameter =  path_of_CModel + '/' + 'model_parameter'
    cnn_model = torch.load( path_of_CModel_model, map_location=device )
    cnn_model.load_state_dict( torch.load(path_of_CModel_parameter, map_location=device) )
    trace_model = torch.jit.trace(cnn_model, torch.rand(1, 2, 201, 3) )
    cnn_model_list.append( trace_model )


# choose some balls

for index in range(11, 12):
# for index in range( position_list.shape[0] ):
    if np.linalg.norm( position_list[index, :], ord=2 ) - (l_1 + l_2) > 1e-10:
        print('{}. ball can not be reached'.format( index ))
    else:
        path_of_folder = path_root + '/' + 'Ball_' + str( index )
        
        # build the robot arm
        RealRobot = Robot(frontend, dof_list, model_num, model_den,
                          model_num_order, model_den_order, model_ndelay_list,
                          inverse_model_num, inverse_model_den, order_num_list, order_den_list,
                          ndelay_list, anchor_ago_list, anchor_ant_list, strategy_list, pid_list,
                          delta_d_list, delta_y_list, delta_w_list, delta_ini_list,
                          pressure_min, pressure_max,
                          A_list, A_bias, cnn_model_list)

        RealRobot.PressureInitialization()

        trajectory_history, trajectory_real, trajectory_NN, p_in_cylinder,\
        v_in_cylinder, p_to_check, ff, fb = RealRobot.HitBall( angle_initial=np.array( frontend.latest().get_positions() ),
                                                        RobotGeometry=RobotGeometry,
                                                        target=RobotGeometry.EndToAngle(position_list[index]),  # target in joint space
                                                        T_go=(time_list[index] / 0.01 - (int(time_list[index] / 0.01)%2)) * 0.01, 
                                                        T_back=1 )
                          
        # save the data
        if os.path.exists( path_of_folder ) == False:
            os.makedirs( path_of_folder )

        path_of_data = path_of_folder + '/' + 'data'
        file = open(path_of_data, 'wb')
        pickle.dump(trajectory_history, file, -1) 
        pickle.dump(trajectory_real, file, -1)
        pickle.dump(trajectory_NN, file, -1)
        pickle.dump(p_in_cylinder, file, -1)
        pickle.dump(v_in_cylinder, file, -1)
        pickle.dump(p_to_check, file, -1)
        pickle.dump(ff, file, -1)
        pickle.dump(fb, file, -1)
        pickle.dump(position_list[index], file, -1)
        pickle.dump(time_list[index], file, -1)
        file.close()
        # return to the initial posture

        # RealRobot.AngleInitialization( angle_initial_read , mode_name='Together')
        # RealRobot.PressureInitialization()
        
