'''
This script is used to control a single joint
'''
import time
import math
import os
import numpy as np
import o80
import o80_pam
from RealJoint import Joint
from RealRobotGeometry import RobotGeometry
import pickle5 as pickle
from RealGenerateMatrices import Filter
import matplotlib.pyplot as plt
# %%
def FulfillZeros( a ):
    b = np.zeros(  [len(a), len(max(a, key = lambda x: len(x)))]  )
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b
# %% read data of balls
path_of_file = "/home/hao/Desktop/Hao/" + 'BallsData' + '.txt'
# path_of_file = "/home/hao/Desktop/Hao/data/" + "F_SmoothSignal_1_a__i_40.txt"
file = open(path_of_file, 'rb')
time_list = pickle.load(file)
position_list = pickle.load(file)
velocity_list = pickle.load(file)
file.close()
# %% Calibration
l_1 = 0.4
l_2 = 0.38

position_mean = np.mean( position_list, axis=0 )
# offset angles of upright posture
offset = np.array( [2.94397627, -0.078539855235, -0.06333859293225] )
# angles of initial posture
angle_initial_ref = np.array( [2.94397627, -0.605516948, -0.5890489142699] )
# anchor angles
angle_anchor_ref = np.array( [ 2.94397627, -1.452987321865, -0.87660612618] )

angle_initial = angle_initial_ref - offset
angle_anchor = angle_anchor_ref - offset

Geometry = RobotGeometry(initial_posture=angle_initial)

(_, position_anchor) = Geometry.AngleToEnd(angle_anchor, frame='Cartesian')
position_error = position_anchor - position_mean
position_list = position_list + position_error  # have been calibrated
# %% constant variables
# connect to the simulation
frontend = o80_pam.FrontEnd("real_robot")

# sampling frequency of the desired trajectory 
fs = 100
# how many iterations
number_iteration = 40
# frequency of the backend
frequency = 500.0
# %% constant variables
# the list for all dofs
dof_list = [0, 1, 2, 3]
dof = 1
# covariance parameter for each dof
delta_d_list = np.array([1e-9, 1e-11, 1e-8, 1e-9]) 
delta_y_list = np.array([1e-3, 1e-3, 1e-3, 1e-3])
delta_w_list = np.array([1e-9, 1e-11, 1e-8, 1e-9])
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
# the set point pressure for each dof
anchor_ago_list = np.array([17500, 18500, 16000, 15000])
anchor_ant_list = np.array([17500, 18500, 16000, 15000])
# the strategy code for each dof
strategy_list = np.array([2, 2, 2, 2])
# parameters of PID controller for each dof
pid_list = [[-3.505924158687806e+04, -3.484022215671791e+05 / 5, -5.665386729745434e+02],
            [8.228984656729296e+04, 1.304087541343074e+04, 4.841489121599795e+02],
            [-36752.24956301624, -246064.5612272051 / 10, -531.2866756516057],
            [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]]
pid_list = FulfillZeros( pid_list )
# constraints for each muscle
pressure_min = [-4500, -6500, -6000, -7000]
pressure_max = [4500, 6500, 6000, 7000]

weight_list = [(1.0, 1.0),
               (1.0, 1.0),
               (1.0, 1.0),
               (1.0, 1.0)]
# %%
root = '/home/hao/Desktop/HitBall/TestSignal'
root_file = root + '/' + 'CraggedSignal.txt'
fid = open(root_file)
time_length = int(fid.readline().strip())
N = int(fid.readline().strip())
y_ = np.zeros(N)
for i in range(N):
    y_[i] = float(fid.readline().strip())

y_ = y_/max(y_)
joint_range = np.array([60, 30, 30, 60])*math.pi/180
y_des = y_ * joint_range.reshape(-1, 1)
t_stamp = np.array(range(N)) * 0.01
# fig = plt.figure(figsize=(16, 16))
# for i in range(4):
#     plt.plot(t_stamp, y_des[i, :]*180/math.pi, label=r'dof {}'.format(i))
# plt.legend(ncol=4)
# plt.show()
# %%
GLOBAL_INITIAL = np.array([0.000000, -0.514884, -0.513349, -0.172187])
for dof in range(4):   
    Joint_0 = Joint(frontend, dof, anchor_ago_list[dof], anchor_ant_list[dof], 
                    inverse_model_num[dof], inverse_model_den[dof],
                    order_num_list[dof], order_den_list[dof], ndelay_list[dof],
                    pid_list[dof], strategy_list[dof])
    # initialize the pressure
    Joint_0.AngleInitialization( GLOBAL_INITIAL[dof], tolerance=0.1 )
    Joint_0.PressureInitialization( anchor_ago_list[dof], anchor_ant_list[dof] )
    angle_ = np.array(frontend.latest().get_positions())
    angle_read = angle_[dof]
    (u_ago, u_ant, ff) = Joint_0.Feedforward(y_des[dof, :], angle_read)
    Joint_0.Control(y=y_des[dof, :], mode_name='fb+ff', mode_trajectory='ref',
                    ifplot='no', echo='no', u_ago=u_ago, u_ant=u_ant, ff=ff)

    # # only get the feedforward control without exciting the simulation/real system
    # # u is the basic presssure and ff is the feedforward control
    # (u_ago, u_ant, ff) = Joint_0.Feedforward(y_desired, angle_initial)
    # # dof, y_abs, dim, pressure_min, pressure_max, num, den, order_num, order_den, ndelay)
    # # use the parameters of model *not* inverse model
    # F = Filter(dof, y_desired, y_desired.shape[0],
    #         pressure_min[dof], pressure_max[dof], 
    #         model_num[dof], model_den[dof],
    #         model_num_order[dof], model_den_order[dof],
    #         model_ndelay_list[dof], angle_, 
    #         total_iteration=number_iteration)
    # # generate the matrices for the kalman filter
    # F.GenerateGlobalMatrix(mode_name='none')

    # y_history = []
    # ff_history = []
    # disturbance_history = []
    # P_history = []
    # fb_history = []
    # ago_history = []
    # ant_history = []
    # d_lifted_history = []
    # P_lifted_history = []

    # y_history.append( np.copy( y_desired ) )

    # P = F.GenerateCovMatrix(delta_d_list[dof],
    #                         delta_y_list[dof], 
    #                         delta_w_list[dof], 
    #                         delta_ini_list[dof])

    # disturbance = np.zeros(len(u_ago)).reshape(-1,1)

    # for i in range(number_iteration):

    #     ff_history.append( np.copy( ff ) )

    #     print("number of iteration: {}".format(i))
    #     # read the output of the simulation/real system
    #     # y is the absolute measured angle 
    #     # y should have one more point than u and y_desired
    #     print("begin to measure...")
    #     (y, fb, obs_ago, obs_ant) = Joint_0.Control(y_desired, 
    #                                                 mode_name="ff", 
    #                                                 ifplot="no", 
    #                                                 u_ago=u_ago, 
    #                                                 u_ant=u_ant,
    #                                                 ff=ff, 
    #                                                 echo="yes")
    #     print("...measurement completed")
    #     # calculate the variables for the kalman filter
    #     # ff + u is used to reach the absolute angle
    #     # P and disturbance will be updated using Kalman filter
    #     print("begin to optimize...")
    #     (ff, disturbance, P, d_lifted, P_lifted) = F.Optimization(y, ff, 
    #                                                                 disturbance, 
    #                                                                 P, number_iteration=i,
    #                                                                 weight=weight_list[dof],
    #                                                                 mode_name='none' )
    #     print("...optimization completed")
    #     # record all the results of each iteration
    #     fb_history.append( np.copy(fb) )
    #     ago_history.append( np.copy(obs_ago) )
    #     ant_history.append( np.copy(obs_ant) )
    #     d_lifted_history.append( np.copy( d_lifted ) )
    #     P_lifted_history.append( np.copy( P_lifted ) )
    #     disturbance_history.append( np.copy( disturbance) )
    #     y_history.append( np.copy( y ) )
    #     P_history.append( np.copy( P ) )
    #     # set the same initial angle for the next iteration
    #     print("begin to initialize...")
    #     Joint_0.AngleInitialization( GLOBAL_INITIAL[dof], tolerance=0.2 )
    #     Joint_0.PressureInitialization( anchor_ago_list[dof], anchor_ant_list[dof] )
    #     print("...initialization completed")
    # # see the final result
    # (u_ago, u_ant, ff_pid) = Joint_0.Feedforward(y_desired, angle_initial)
    # (y_pid, fb_pid, obs_ago, obs_ant) = Joint_0.Control(y_desired, mode_name="ff+fb", ifplot="no", u_ago=u_ago, u_ant=u_ant, ff=ff_pid, echo="yes")
    # Joint_0.AngleInitialization( GLOBAL_INITIAL[dof], tolerance=0.2 )
    # Joint_0.PressureInitialization( anchor_ago_list[dof], anchor_ant_list[dof] )
    # # %%
    # def mkdir(path):
    #     folder = os.path.exists(path)
    #     if not folder:
    #         os.makedirs(path)
    # # choose one trajectory to train the model
    # path_of_folder = "/home/hao/Desktop/HitBall/RSS2022"
    # mkdir(path_of_folder)
    # path_of_file = path_of_folder + '/' + "Comparison_PID_ILC"
    # file = open(path_of_file, 'wb')
    # pickle.dump(t_stamp, file, -1) # time stamp for x-axis
    # pickle.dump(t_list, file, -1)
    # pickle.dump(angle_read, file, -1)  
    # pickle.dump(y_history, file, -1)
    # pickle.dump(ff_history, file, -1)
    # pickle.dump(fb_history, file, -1)
    # pickle.dump(ago_history, file, -1)
    # pickle.dump(ant_history, file, -1)
    # pickle.dump(disturbance_history, file, -1)
    # pickle.dump(P_history, file, -1)
    # pickle.dump(d_lifted_history, file, -1)
    # pickle.dump(P_lifted_history, file, -1)
    # pickle.dump(ff_pid, file, -1)
    # pickle.dump(fb_pid, file, -1)
    # pickle.dump(y_pid, file, -1)
    # file.close()


