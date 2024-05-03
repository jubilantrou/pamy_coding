'''
frequency_frontend for controlling the robot
'''
import math
import numpy as np
import o80
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from RealJoint import Joint
from RealGenerateMatrices import Filter
import time
from threading import Thread
import multiprocessing as mp
from FastLCNN import LCNN
import torch
import scipy
from OCO_plots import *
import PAMY_CONFIG
from RealRobotGeometry import RobotGeometry

class Robot:
    def __init__(self, frontend, dof_list, model_num, model_den,
                 model_num_order, model_den_order, model_ndelay_list,
                 inverse_model_num, inverse_model_den, order_num_list, order_den_list,
                 ndelay_list, anchor_ago_list, anchor_ant_list, strategy_list, pid_list,
                 delta_d_list, delta_y_list, delta_w_list, delta_ini_list, 
                 pressure_min, pressure_max, weight_list,
                 A_list=None, A_bias=None, cnn_model_list=None):
        '''
        This class is used to build the robot.
        frontend: connect to the backend of pamy
        dof_list: the list of dofs that you want to contol
        model_num, model_den, model_ndelay_list: parameters that describe the discrete linear forward 
                                                 model of each dof
        inverse_model_num, inverse_model_den, ndelay_list: parameters that describe the discrete linear
                                                           inverse model of each dof
        ''' 
        # connect to the backend
        self.frontend = frontend
        # all dofs
        self.dof_list = dof_list
        # linear model
        self.model_num = model_num
        self.model_den = model_den
        self.model_num_order = model_num_order
        self.model_den_order = model_den_order
        self.model_ndelay_list = model_ndelay_list
        # inverse model
        self.inverse_model_num = inverse_model_num
        self.inverse_model_den = inverse_model_den
        self.order_num_list = order_num_list
        self.order_den_list = order_den_list
        self.ndelay_list = ndelay_list
        # anchor pressures for each dof
        self.anchor_ago_list = anchor_ago_list
        self.anchor_ant_list = anchor_ant_list
        # control strategies
        self.strategy_list = strategy_list
        # pid controller to return to initial posture
        self.pid_list = pid_list
        # covariance parameters for ILC
        self.delta_d_list = delta_d_list
        self.delta_y_list = delta_y_list
        self.delta_w_list = delta_w_list
        self.delta_ini_list = delta_ini_list
        # set bounds for optimization
        self.pressure_min = pressure_min
        self.pressure_max = pressure_max
        self.weight_list = weight_list
        # define four joints
        self.Joint_list = []
        '''
        Each dof can be seen as independent.
        '''
        for dof in self.dof_list:
            Joint_temp = Joint(self.frontend, self.dof_list[dof], self.anchor_ago_list[dof], self.anchor_ant_list[dof], 
                               self.inverse_model_num[dof, :], self.inverse_model_den[dof, :],
                               self.order_num_list[dof], order_den_list[dof], ndelay_list[dof], 
                               self.pid_list[dof, :], self.strategy_list[dof])
            self.Joint_list.append( Joint_temp )
        # frequency and period
        frequency_backend = 500
        frequency_frontend = 100 # default is 50
        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        self.step_size = 1 / frequency_frontend
        self.iterations_per_command = int( period_frontend / period_backend )  # sychronize the frontend and the backend
        # pid for tracking
        '''
        the original one
        '''
        # self.pid_for_tracking = np.array([[-13000, 0, -300],
        #                                   [ 80000, 0, 300],
        #                                   [ -5000, -8000, -100],
        #                                   [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])
        # self.pid_for_tracking = np.array([[-13000, 0, -300],
        #                                   [-30000, 0, -300],
        #                                   [-5000, -8000, -100],
        #                                   [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])
        # self.pid_for_tracking = 0.6*np.array([[-10000,  0, -260],
        #                                       [-10500,  0, -525],
        #                                       [-18000,  0, -875],
        #                                       [-12000, 0, -1000]]) # tuned PD for the simulator
        self.pid_for_tracking = np.array([[0,  0, 0],
                                          [0,  0, 0],
                                          [0,  0, 0],
                                          [0,  0, 0]]) # w/o PD
        # self.pid_for_tracking = 0.6*np.array([[-4420,  0, -276.25],
        #                                       [-7040,  0, -733.04],
        #                                       [-12080,  0, -419.78],
        #                                       [0,  0, 0]]) # PD for the real robot
        self.m_4_y = 2
        self.n_4_u = 2
        # self.lqr_k = np.squeeze(scipy.io.loadmat('/home/mtian/Desktop/lqr_K.mat')['K'])
        self.lqi_k = scipy.io.loadmat('/home/mtian/Desktop/K_i.mat')['K_i']
        self.lqi_k_tracking = scipy.io.loadmat('/home/mtian/Desktop/K_i_tracking.mat')['K_i']
        self.lqr_k = scipy.io.loadmat('/home/mtian/Desktop/lqr_K.mat')['K_inf']
        # NN
        # NN
        self.A_list = A_list 
        self.A_bias = A_bias
        self.cnn_model_list = cnn_model_list
        # length should be long enough
        self.trajectory_temp = np.zeros((4, 1000))
        self.trajectory_history = []
        self.trajectory_real = np.zeros((4, 1000))
        self.p_in_cylinder = np.zeros((3, 1000))
        self.v_in_cylinder = np.zeros((3, 1000))
        self.a_in_cylinder = np.zeros((3, 1000))
        self.p_to_check = np.zeros((3, 1000))
        self.ff = np.zeros((4, 1000))
        self.fb = np.zeros((4, 1000))
    
    def GetOptimizer(self, angle_initial, total_iteration=40, mode_name='none'):
        '''
        y_desired:       relative desired trajectories
        t_stamp:         time stamp for the trajectories
        angle_initial:   initial angles
        total_iteration: the number of iterations for ILC
        mode_name:       training mode
                         'none' - train without feedback inputs
                         'pd'   - train with pd controller

        y_desired + angle_initial = absolute trajectory
        '''
        self.Optimizer_list = []
        for dof in self.dof_list:
            Optimizer = Filter(dof, self.y_desired[dof, :], self.v_desired[dof, :], self.a_desired[dof, :], self.j_desired[dof, :], self.y_desired.shape[1],
                                    self.pressure_min[dof], self.pressure_max[dof], self.model_num[dof, :], self.model_den[dof, :], self.model_num_order[dof],
                                    self.model_den_order[dof], self.model_ndelay_list[dof], angle_initial, total_iteration=total_iteration)
            Optimizer.GenerateGlobalMatrix(mode_name)
            self.Optimizer_list.append(Optimizer)
    
    
    def get_Xi(self, h_in_left=100, h_in_right=100):

        def get_compensated_data(data=None):
            # I = np.zeros((3, h_in_left))
            I_left = np.tile(data[:, 0].reshape(-1, 1), (1, h_in_left))
            I_right = np.tile(data[:, -1].reshape(-1, 1), (1, h_in_right))
            new_data = np.hstack((I_left, data, I_right))
            return new_data
          
        y_comp = get_compensated_data(self.y_desired[0:3, :])
        Xi = np.zeros((self.y_desired.shape[1], 3*(h_in_left+h_in_right+1)+1))
    
        for k in range(self.y_desired.shape[1]):
            Xi[k, :] = np.hstack((y_comp[0, k:k+2*h_in_left+1], y_comp[1, k:k+2*h_in_left+1], y_comp[2, k:k+2*h_in_left+1], 1))

        return Xi
    
    
    def GetOptimizer_convex(self, angle_initial, h=100, nr_channel=1, mode_name='none', coupling='no', learning_mode='b', Bu_mode='calculation'):
        '''
        y_desired:       relative desired trajectories
        t_stamp:         time stamp for the trajectories
        angle_initial:   initial angles
        total_iteration: the number of iterations for ILC
        mode_name:       training mode
                            'none' - train without feedback inputs
                            'pd'   - train with pd controller

        y_desired + angle_initial = absolute trajectory
        '''
        # self.Xi            = self.get_Xi(h_in_left=h, h_in_right=h)
        self.hessian_list  = []
        self.gradient_list = []
        self.part2_list    = []
        self.O_list        = []
        self.Xi_list       = []
        
        for dof in self.dof_list:
            O = Filter(dof, self.y_desired[dof, :], self.v_desired[dof, :], self.a_desired[dof, :], self.j_desired[dof, :], self.y_desired.shape[1],
                       self.pressure_min[dof], self.pressure_max[dof], self.model_num[dof, :], self.model_den[dof, :],
                       self.model_num_order[dof], self.model_den_order[dof], self.model_ndelay_list[dof], angle_initial)
            # TODO                                    
            O.GenerateGlobalMatrix_convex(h=h, nr_channel=nr_channel, mode_name=mode_name, Bu_mode=Bu_mode)
            self.O_list.append(O)

            # if learning_mode == 'u':
            #     gradient = O.Bu
            #     hessian  = (O.Bu).T@(O.Bu)
            #     part2    = gradient.T@(O.y_des) # - O.Bdes@O.y_des_bar)
            # elif learning_mode == 'b':
            #     if coupling == 'yes':
            #         gradient = O.Bu@self.Xi
            #         hessian  = (O.Bu@self.Xi).T@(O.Bu@self.Xi)
            #         part2    = gradient.T@(O.y_des) # - O.Bdes@O.y_des_bar)
            #         Xi       = self.Xi
            #     elif coupling == 'no':         
            #         gradient = O.Bu@O.Xi
            #         hessian  = (O.Bu@O.Xi).T@(O.Bu@O.Xi)
            #         part2    = gradient.T@(O.y_des) # - O.Bdes@O.y_des_bar)
            #         Xi       = O.Xi
           
            # self.gradient_list.append(gradient)
            # self.part2_list.append(part2)
            # self.hessian_list.append(hessian)
            # self.Xi_list.append(Xi)
            
    def ImportTrajectory(self, y_desired, t_stamp):
        def get_difference(x):
            y = np.zeros(x.shape)
            for i in range(1, y.shape[1]):
                y[:, i] = (x[:, i]-x[:, i-1])/(t_stamp[i]-t_stamp[i-1])
            return y
        '''
        This function is used to update the desired trjectories of the Robot.
        The desired trajectories should be relative trajectories.
        '''
        self.y_desired = np.copy(y_desired)
        self.v_desired = get_difference(self.y_desired)
        self.a_desired = get_difference(self.v_desired)
        self.j_desired = get_difference(self.a_desired)
        self.t_stamp = t_stamp
        
    def Feedforward(self, y_list=None):
        '''
        here y_list should be the relative angle
        '''
        if y_list is None:
           y_list = np.copy( self.y_desired )
        # basic pressure for ago
        u_ago = np.array([])
        # basic pressure for ant
        u_ant = np.array([])
        # feedforward control
        ff = np.array([])
        for dof in self.dof_list:
            (u_ago_temp, u_ant_temp, ff_temp) = self.Joint_list[dof].Feedforward(y_list[dof, :])
            u_ago = np.append(u_ago, u_ago_temp)
            u_ant = np.append(u_ant, u_ant_temp)
            ff = np.append(ff, ff_temp)
        ff = ff.reshape(len(self.dof_list), -1)
        u_ago = u_ago.reshape(len(self.dof_list), -1)
        u_ant = u_ant.reshape(len(self.dof_list), -1)
        return(u_ago, u_ant, ff)

    def Control(self, y_list=None, mode_name_list=["fb+ff", "fb+ff", "fb+ff", "fb+ff"], 
                mode_trajectory="ref",
                frequency_frontend=100, frequency_backend=500,
                ifplot="yes", u_ago=None, u_ant=None, ff=None, echo="no", controller='pid', trainable_fb=None, device=None, dim_fb=None):
        # import the reference trajectory
        if y_list is None:
            y_list = np.copy( self.y_desired )
        # generate the corresponding time stamp
        t_stamp_u = np.linspace(0, (y_list.shape[1] - 1) / frequency_frontend, y_list.shape[1] )

        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )  # sychronize the frontend and the backend
        
        # read the actual current states
        theta = self.frontend.latest().get_positions()
        theta = np.array( theta )
        # theta_dot = self.frontend.latest().get_velocities()
        # theta_dot = np.array( theta_dot )
        # pressures = self.frontend.latest().get_observed_pressures()
        # pressures = np.array( pressures )
        # the actual ago pressures and ant pressures
        # pressure_ago = pressures[:, 0]
        # pressure_ant = pressures[:, 1]
        # reference trajectory or absolute trajectory
        # reference trajectory for tracking 
        # absolute trajectory for initial posture
        if mode_trajectory == "ref":
            angle_initial = theta
            for dof in self.dof_list:
                print("{}. initial angle is: {:.2f} degree".format(dof, angle_initial[dof] * 180 / math.pi))
        elif mode_trajectory == "abs":
            angle_initial = np.zeros(len(self.dof_list))
        # if u_ago or u_ant or ff is not specified then calculate the feedforward
        if (u_ago is None) or (u_ant is None) or (ff is None):
            (u_ago, u_ant, ff) = self.Feedforward( self.y_desired, angle_initial )

        fb = np.array([])
        
        pid = np.copy( self.pid_list )
        if controller == 'pd':
            pid = np.copy( self.pid_for_tracking )

        angle_delta_pre = np.zeros( len(self.dof_list) ).reshape(len(self.dof_list), -1)
        res_i = 0

        # read the current iteration
        iteration_reference = self.frontend.latest().get_iteration()  
        # set the beginning iteration number
        iteration_begin = iteration_reference + 500

        iteration = iteration_begin

        self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                  o80.Iteration(iteration_begin-1),
                                  o80.Mode.QUEUE)
        
        self.frontend.pulse()

        # theta = self.frontend.latest().get_positions()
        # theta = np.array( theta )
        # if mode_trajectory == "ref":
        #     angle_initial = theta
        #     for dof in self.dof_list:
        #         print("{}. initial angle is: {:.2f} degree".format(dof, angle_initial[dof] * 180 / math.pi))
        # elif mode_trajectory == "abs":
        #     angle_initial = np.zeros(len(self.dof_list))
        # if (u_ago is None) or (u_ant is None) or (ff is None):
        #     (u_ago, u_ant, ff) = self.Feedforward( self.y_desired, angle_initial )

        # fb_inputs = [[0.0]*dim_fb]*3
        # fb_datasets = [[],[],[]]
        fb_datasets = None

        # fb inputs from LQI
        # lqr = np.copy( self.lqi_k_tracking )
        # m = self.m_4_y
        # n = self.n_4_u
        # x_in = np.zeros((3*(m+n),1))
        # x_e = 0.01*(np.array([[0],[0],[0]]) - x_in[(3*m-3):3*m])
        # cur_states = [0.0] * (3*m)
        # delta_pressures = [0.0] * (3*n)

        # fb inputs from LQR
        lqr = np.copy( self.lqr_k )
        m = self.m_4_y
        n = self.n_4_u
        cur_states = [0.0] * (3*m)
        delta_pressures = [0.0] * (3*n)
        des_states = [0.0] * (3*m)
        des_pressures = [0.0] * (3*n)
        x_in = np.concatenate((np.array(cur_states).reshape(3*m,1),np.array(delta_pressures).reshape(3*n,1)),axis=0)
        x_des = np.concatenate((np.array(des_states).reshape(3*m,1),np.array(des_pressures).reshape(3*n,1)),axis=0)

        record_lqr = np.array([])
        # how mang steps to track
        for i in range( y_list.shape[1] ):
            # all following vectors must be column vectors
            angle_delta = (y_list[:, i] + angle_initial - theta).reshape(len(self.dof_list), -1)

            # fb_outputs = np.array([0.0]*4)
            # for i_t in range(3):
            #     fb_inputs[i_t].pop(0)
            #     fb_inputs[i_t].append(angle_delta[i_t,0])
            #     temp_input = torch.tensor(np.array(fb_inputs[i_t]), dtype=float).view(-1).to(device)
            #     fb_datasets[i_t].append(temp_input)
            #     if trainable_fb[i_t] is None:
            #         continue
            #     trainable_fb[i_t].eval()
            #     try:
            #         fb_outputs[i_t] = trainable_fb[i_t](temp_input).cpu().detach().numpy().flatten()
            #     except:
            #         fb_outputs[i_t] = trainable_fb[i_t](temp_input.float()).cpu().detach().numpy().flatten()
            # fb_outputs = fb_outputs.reshape(len(self.dof_list), -1)

            res_d = ( angle_delta - angle_delta_pre ) / t
            res_i += angle_delta * t

            feedback = pid[:, 0].reshape(len(self.dof_list), -1) * angle_delta\
                     + pid[:, 1].reshape(len(self.dof_list), -1) * res_i\
                     + pid[:, 2].reshape(len(self.dof_list), -1) * res_d
            
            angle_delta_pre = np.copy( angle_delta )

            if i == 0:
                fb = np.copy( feedback )
                # fb = np.copy( fb_outputs )
            else:
                fb = np.hstack((fb, feedback))
                # fb = np.hstack((fb, fb_outputs))

            pressure_ago = np.array([], dtype=int)
            pressure_ant = np.array([], dtype=int)
        
            # x_aug = np.concatenate((x_in,x_e),axis=0)
            # fb_lqi = -lqr@x_aug

            fb_lqr = -lqr@(x_in-x_des)
            if i == 0:
                record_lqr = np.copy( fb_lqr )
            else:
                record_lqr = np.hstack((record_lqr, fb_lqr))
            
            diff_store = np.zeros((3,1))
            for dof in self.dof_list:
                if mode_name_list[dof] == "ff":
                    diff = ff[dof, i]
                elif mode_name_list[dof] == "fb":
                    diff = fb[dof, i]
                elif mode_name_list[dof] == "fb+ff" or mode_name_list[dof] == "ff+fb":
                    # if dof<3:
                    #     diff = ff[dof, i] + fb[dof, i] + fb_lqr[dof, 0]
                    #     diff_store[dof] = diff
                    # else:
                    #     diff = ff[dof, i] + fb[dof, i]
                    diff = ff[dof, i] + fb[dof, i]

                if self.strategy_list[dof] == 2:
                    if diff > 0:
                        pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] + diff ))
                        pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] ))
                    elif diff < 0:
                        pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] ))
                        pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] - diff ))
                    else:
                        pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] ))
                        pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] ))
                elif self.strategy_list[dof] == 1:
                    pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] + diff ))
                    pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] - diff ))
            
            # do not control the last dof
            pressure_ago[3] = self.anchor_ago_list[3]
            pressure_ant[3] = self.anchor_ant_list[3]

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
    
            self.frontend.pulse()

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)

            self.frontend.pulse_and_wait()
            # update the angles
            theta = self.frontend.latest().get_positions()
            theta = np.array( theta )
            # update the angular velocities
            # theta_dot = observation.get_velocities()
            # theta_dot = np.array( theta_dot )

            iteration += iterations_per_command

            # if True:
            #     update_states = theta.reshape(-1)-angle_initial.reshape(-1)
            #     [cur_states.append(update_states.tolist()[i]) for i in range(3)]
            #     [cur_states.pop(0) for i in range(3)]
            #     [delta_pressures.append(diff_store.reshape(-1).tolist()[i]) for i in range(3)]
            #     [delta_pressures.pop(0) for i in range(3)]
            #     x_in = np.concatenate((np.array(cur_states).reshape(3*m,1),np.array(delta_pressures).reshape(3*n,1)),axis=0)
            #     x_e = x_e + 0.01*(y_list[0:3,i].reshape(3,1) - x_in[(3*m-3):3*m])

            if True:
                update_states = theta.reshape(-1)-PAMY_CONFIG.GLOBAL_INITIAL
                [cur_states.append(update_states.tolist()[j_temp]) for j_temp in range(3)]
                [cur_states.pop(0) for j_temp in range(3)]
                [delta_pressures.append(diff_store.reshape(-1).tolist()[j_temp]) for j_temp in range(3)]
                [delta_pressures.pop(0) for j_temp in range(3)]
                x_in = np.concatenate((np.array(cur_states).reshape(3*m,1),np.array(delta_pressures).reshape(3*n,1)),axis=0)
                [des_states.append(y_list[:,i].tolist()[j_temp]) for j_temp in range(3)]
                [des_states.pop(0) for j_temp in range(3)]
                [des_pressures.append(ff[j_temp, i]) for j_temp in range(3)]
                [des_pressures.pop(0) for j_temp in range(3)]
                x_des = np.concatenate((np.array(des_states).reshape(3*m,1),np.array(des_pressures).reshape(3*n,1)),axis=0)

        
        iteration_end = iteration

        # print(record_lqr.shape)
        # print('ranges of LQR inputs:')
        # print('dof 0: {} ~ {}'.format(min(record_lqr[0,:]),max(record_lqr[0,:])))
        # print('dof 1: {} ~ {}'.format(min(record_lqr[1,:]),max(record_lqr[1,:])))
        # print('dof 2: {} ~ {}'.format(min(record_lqr[2,:]),max(record_lqr[2,:])))
        
        if ifplot == "yes":
            self.PlotFigures(y_list, angle_initial, ff, fb, t_stamp_u, iteration_begin, iteration_end, iterations_per_command)
        
        if echo == "yes":
            position = np.array([])
            iteration = iteration_begin
            pressure_ago = np.array([])
            pressure_ant = np.array([])
            des_pressure_ago = np.array([])
            des_pressure_ant = np.array([])

            while iteration < iteration_end:
                observation = self.frontend.read(iteration)
                obs_position = np.array( observation.get_positions() )
                obs_pressure = np.array(observation.get_observed_pressures())
                des_pressure = np.array(observation.get_desired_pressures())

                pressure_ago = np.append(pressure_ago, obs_pressure[:, 0])
                pressure_ant = np.append(pressure_ant, obs_pressure[:, 1])
                des_pressure_ago = np.append(des_pressure_ago, des_pressure[:, 0])
                des_pressure_ant = np.append(des_pressure_ant, des_pressure[:, 1])

                position = np.append(position, obs_position)
                iteration += iterations_per_command
            
            position = position.reshape(-1, len(self.dof_list)).T
            pressure_ago = pressure_ago.reshape(-1, len(self.dof_list)).T
            pressure_ant = pressure_ant.reshape(-1, len(self.dof_list)).T
            des_pressure_ago = des_pressure_ago.reshape(-1, len(self.dof_list)).T
            des_pressure_ant = des_pressure_ant.reshape(-1, len(self.dof_list)).T

            # observation_prev = self.frontend.read(iteration_begin-iterations_per_command)
            # obs_position_prev = np.array( observation_prev.get_positions() )
            # position = np.hstack([obs_position_prev.reshape(4,1), position[:,:-1]])
            
            # print('final positions:')
            # print(position[:,-1]/math.pi*180)

            return (position, fb, pressure_ago, pressure_ant, des_pressure_ago, des_pressure_ant, fb_datasets)
         
    def PressureInitialization(self, times=1, duration=1):
        for _ in range(times):
            # creating a command locally. The command is *not* sent to the robot yet.
            self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                      o80.Duration_us.seconds(duration),
                                      o80.Mode.QUEUE)
            # sending the command to the robot, and waiting for its completion.
            self.frontend.pulse_and_wait()
            time.sleep(duration)
        # for dof in self.dof_list:
        #     print("the {}. ago/ant pressure is: {:.2f}/{:.2f}".format(dof, pressures[dof, 0], pressures[dof, 1]) )

    def AngleInitialization(self, angle, tolerance_list=[0.1,0.1,0.1,1.0], 
                            frequency_frontend=100, frequency_backend=500):        
        pid = np.copy( self.pid_list )
        tolerance_list = np.array(tolerance_list)
        theta = self.frontend.latest().get_positions()
        res_i = 0
        angle_delta_pre = np.zeros( len(self.dof_list) ).reshape(len(self.dof_list), -1)

        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )

        iteration = self.frontend.latest().get_iteration() + 500  # set the iteration when beginning
        
        self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                  o80.Iteration(iteration-iterations_per_command),
                                  o80.Mode.QUEUE)
        
        self.frontend.pulse()
        # do not consider the last dof
        # while 1:
        while not (abs((theta[0:3] - angle[0:3])*180/math.pi) < tolerance_list[0:3]).all():
            
            # if (abs((theta[0:3] - angle[0:3])*180/math.pi) < tolerance_list[0:3]).all():
            #     time.sleep(0.1)
            #     theta = self.frontend.latest().get_positions()
            #     if (abs((theta[0:3] - angle[0:3])*180/math.pi) < tolerance_list[0:3]).all():
            #         print('current pressures')
            #         print(self.frontend.latest().get_desired_pressures())
            #         break

            # all following vectors must be column vectors
            angle_delta = (angle - theta).reshape(len(self.dof_list), -1)
            res_d = ( angle_delta - angle_delta_pre ) / t
            res_i += angle_delta * t

            feedback = pid[:, 0].reshape(len(self.dof_list), -1) * angle_delta\
                     + pid[:, 1].reshape(len(self.dof_list), -1) * res_i\
                     + pid[:, 2].reshape(len(self.dof_list), -1) * res_d
            
            angle_delta_pre = np.copy( angle_delta )

            pressure_ago = np.array([], dtype=int)
            pressure_ant = np.array([], dtype=int)

            for dof in self.dof_list:
                diff = feedback[dof]
                pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] + diff ))
                pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] - diff ))
            
            # do not control the last dof
            pressure_ago[3] = self.anchor_ago_list[3]
            pressure_ant[3] = self.anchor_ant_list[3]

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
    
            self.frontend.pulse()

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)

            observation = self.frontend.pulse_and_wait()
            # update the angles
            theta = np.array(observation.get_positions())

            iteration += iterations_per_command

        for dof in self.dof_list:
            print("the {}. expected/actual angle: {:.2f}/{:.2f} degree".format(dof, angle[dof] * 180 / math.pi, theta[dof] * 180 / math.pi))
            print("the error of {}. angle: {:.2f} degree".format(dof, (theta[dof] - angle[dof]) * 180 / math.pi))
    
    def PIDTesting(self, choice, amp, t_start, t_duration, frequency_frontend=100, frequency_backend=500):
        '''
        This function gives a step signal to the object to help with finding proper PID parameters, 
        with modification based on the function AngleInitialization() above.
        '''
        pid = np.copy( self.pid_list )

        period_backend = 1.0 / frequency_backend
        period_frontend = 1.0 / frequency_frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )

        self.PressureInitialization()

        iteration = self.frontend.latest().get_iteration() + 200
        iteration_begin = iteration
        self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                  o80.Iteration(iteration-iterations_per_command),
                                  o80.Mode.QUEUE)
        obs_begin = self.frontend.pulse_and_wait()
        theta = obs_begin.get_positions()
        theta_begin = np.copy(theta)
        
        res_i = 0
        angle_delta_pre = 0

        count = 0
        input = np.array([])
        while count <= t_duration*frequency_frontend:
            if count <= t_start*frequency_frontend:
                target = theta_begin[choice]
            else:
                target = theta_begin[choice] + amp
            input = np.append(input, target)
            
            angle_delta = target - theta[choice]
            res_d = ( angle_delta - angle_delta_pre ) / t
            res_i += angle_delta * t
            angle_delta_pre = angle_delta

            feedback = pid[choice, 0] * angle_delta\
                     + pid[choice, 1] * res_i\
                     + pid[choice, 2] * res_d
            
            pressure_ago = [int(self.anchor_ago_list[i]+feedback) if i==choice else self.anchor_ago_list[i] for i in self.dof_list]
            pressure_ant = [int(self.anchor_ant_list[i]-feedback) if i==choice else self.anchor_ant_list[i] for i in self.dof_list]
            
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)    
            self.frontend.pulse()
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)
            observation = self.frontend.pulse_and_wait()

            theta = np.array(observation.get_positions())
            iteration += iterations_per_command
            count += 1

        iteration_end = iteration
            
        time = np.array([])
        position = np.array([])

        iteration = iteration_begin
        while iteration < iteration_end:
            observation = self.frontend.read(iteration)
            obs_position = np.array( observation.get_positions() )
            obs_time = np.array( observation.get_time_stamp() )*1e-9

            position = np.append(position, obs_position[choice])
            time = np.append(time, obs_time)

            iteration += iterations_per_command
        
        time = time-time[0]
        return time, input, position
    
    def LQRTesting(self, amp, t_start, t_duration, frequency_frontend=100, frequency_backend=500):
        lqr = np.copy( self.lqi_k )
        m = self.m_4_y
        n = self.n_4_u

        period_backend = 1.0 / frequency_backend
        period_frontend = 1.0 / frequency_frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )

        # self.PressureInitialization()

        iteration = self.frontend.latest().get_iteration() + 500
        iteration_begin = iteration
        self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                  o80.Iteration(iteration-iterations_per_command),
                                  o80.Mode.QUEUE)
        obs_begin = self.frontend.pulse_and_wait()
        theta = np.array(obs_begin.get_positions())
        theta_begin = np.copy(theta)

        amp = np.array([[0],[60],[40]])/180*math.pi - theta_begin.reshape(4,1)[0:3]

        count = 0
        input = np.array([])
        compute_diff = np.array([])
        x_in = np.zeros((3*(m+n),1))
        x_e = 0.01*(amp - x_in[(3*m-3):3*m])
        cur_states = [0.0] * (3*m)
        delta_pressures = [0.0] * (3*n)
        while count < t_duration*frequency_frontend:
            if count < t_start*frequency_frontend:
                target = theta_begin.reshape(-1)[0:3]
            else:
                target = theta_begin.reshape(-1)[0:3] + amp.reshape(-1)
            input = np.append(input, target.reshape(-1))

            x_aug = np.concatenate((x_in,x_e),axis=0)
            feedback = -lqr@x_aug
            compute_diff = np.append(compute_diff, feedback.reshape(-1))
            
            pressure_ago = [int(self.anchor_ago_list[i]+feedback[i,0]) if i<3 else self.anchor_ago_list[i] for i in self.dof_list]
            pressure_ant = [int(self.anchor_ant_list[i]-feedback[i,0]) if i<3 else self.anchor_ant_list[i] for i in self.dof_list]
            
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)    
            self.frontend.pulse()
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)
            observation = self.frontend.pulse_and_wait()

            theta = np.array(observation.get_positions())
            iteration += iterations_per_command
            count += 1
            
            update_states = theta.reshape(-1)-theta_begin.reshape(-1)
            [cur_states.append(update_states.tolist()[i]) for i in range(3)]
            [cur_states.pop(0) for i in range(3)]
            [delta_pressures.append(feedback.reshape(-1).tolist()[i]) for i in range(3)]
            [delta_pressures.pop(0) for i in range(3)]
            x_in = np.concatenate((np.array(cur_states).reshape(3*m,1),np.array(delta_pressures).reshape(3*n,1)),axis=0)
            x_e = x_e + 0.01*(amp - x_in[(3*m-3):3*m])

        iteration_end = iteration
            
        time = np.array([])
        position = np.array([])

        iteration = iteration_begin
        while iteration < iteration_end:
            observation = self.frontend.read(iteration)
            obs_position = np.array( observation.get_positions() )
            obs_time = np.array( observation.get_time_stamp() )*1e-9

            position = np.append(position, obs_position.reshape(-1)[0:3])
            time = np.append(time, obs_time)

            iteration += iterations_per_command
        
        time = time-time[0]
        
        return time, input, position, compute_diff, theta_begin.reshape(-1)
    
    def LQRTrackingTesting(self, amp, frequency_frontend=100, frequency_backend=500):
        lqr = np.copy( self.lqi_k_tracking )
        m = self.m_4_y
        n = self.n_4_u

        period_backend = 1.0 / frequency_backend
        period_frontend = 1.0 / frequency_frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )

        iteration = self.frontend.latest().get_iteration() + 500
        iteration_begin = iteration
        self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                  o80.Iteration(iteration-iterations_per_command),
                                  o80.Mode.QUEUE)
        obs_begin = self.frontend.pulse_and_wait()
        theta = np.array(obs_begin.get_positions())
        theta_begin = np.copy(theta)

        count = 0
        input = np.array([])
        compute_diff = np.array([])
        x_in = np.zeros((3*(m+n),1))
        x_e = 0.01*(np.array([[0],[0],[0]]) - x_in[(3*m-3):3*m])
        cur_states = [0.0] * (3*m)
        delta_pressures = [0.0] * (3*n)
        while count < amp.shape[1]:
            input = np.append(input, theta_begin.reshape(-1)[0:3] + amp[0:3,count].reshape(-1))

            x_aug = np.concatenate((x_in,x_e),axis=0)
            feedback = -lqr@x_aug
            compute_diff = np.append(compute_diff, feedback.reshape(-1))
            
            pressure_ago = [int(self.anchor_ago_list[i]+feedback[i,0]) if i<3 else self.anchor_ago_list[i] for i in self.dof_list]
            pressure_ant = [int(self.anchor_ant_list[i]-feedback[i,0]) if i<3 else self.anchor_ant_list[i] for i in self.dof_list]
            
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)    
            self.frontend.pulse()
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)
            observation = self.frontend.pulse_and_wait()

            theta = np.array(observation.get_positions())
            iteration += iterations_per_command
            count += 1
            
            if count < amp.shape[1]:
                update_states = theta.reshape(-1)-theta_begin.reshape(-1)
                [cur_states.append(update_states.tolist()[i]) for i in range(3)]
                [cur_states.pop(0) for i in range(3)]
                [delta_pressures.append(feedback.reshape(-1).tolist()[i]) for i in range(3)]
                [delta_pressures.pop(0) for i in range(3)]
                x_in = np.concatenate((np.array(cur_states).reshape(3*m,1),np.array(delta_pressures).reshape(3*n,1)),axis=0)
                x_e = x_e + 0.01*(amp[0:3,count].reshape(3,1) - x_in[(3*m-3):3*m])

        iteration_end = iteration
            
        time = np.array([])
        position = np.array([])

        iteration = iteration_begin
        while iteration < iteration_end:
            observation = self.frontend.read(iteration)
            obs_position = np.array( observation.get_positions() )
            obs_time = np.array( observation.get_time_stamp() )*1e-9

            position = np.append(position, obs_position.reshape(-1)[0:3])
            time = np.append(time, obs_time)

            iteration += iterations_per_command
        
        time = time-time[0]
        
        return time, input, position, compute_diff, theta_begin.reshape(-1)
    
    def LQRTestingFollowup(self, tar, t_duration, frequency_frontend=100, frequency_backend=500):
        lqr = np.copy( self.lqr_k )
        m = self.m_4_y
        n = self.n_4_u

        period_backend = 1.0 / frequency_backend
        period_frontend = 1.0 / frequency_frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )

        iteration = self.frontend.latest().get_iteration() + 500
        iteration_begin = iteration
        pressure_read = np.array(self.frontend.latest().get_observed_pressures())
        self.frontend.add_command(pressure_read[:,0], pressure_read[:,1],
                                  o80.Iteration(iteration-iterations_per_command),
                                  o80.Mode.QUEUE)
        obs_begin = self.frontend.pulse_and_wait()
        theta = np.array(obs_begin.get_positions())

        count = 0
        input = np.array([])
        compute_diff = np.array([])

        cur_states = [0.0] * (3*m)
        delta_pressures = [0.0] * (3*n)
        cur_states[(3*m-3):3*m] = (theta.reshape(-1)-tar).tolist()[0:3]
        delta_pressures[(3*n-3):3*n] = (pressure_read[:,0].reshape(-1)-self.anchor_ago_list).tolist()[0:3]
        x_in = np.concatenate((np.array(cur_states).reshape(3*m,1),np.array(delta_pressures).reshape(3*n,1)),axis=0)

        while count < t_duration*frequency_frontend:
            # print(count)

            target = tar[0:3]
            input = np.append(input, target.reshape(-1))

            feedback = -lqr@x_in
            compute_diff = np.append(compute_diff, feedback.reshape(-1))
            # print(x_in)
            # print(feedback)
            
            pressure_ago = [int(self.anchor_ago_list[i]+feedback[i,0]) if i<3 else self.anchor_ago_list[i] for i in self.dof_list]
            pressure_ant = [int(self.anchor_ant_list[i]-feedback[i,0]) if i<3 else self.anchor_ant_list[i] for i in self.dof_list]
            
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)    
            self.frontend.pulse()
            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)
            observation = self.frontend.pulse_and_wait()

            theta = np.array(observation.get_positions())
            iteration += iterations_per_command
            count += 1
            
            update_states = theta.reshape(-1)-tar
            [cur_states.append(update_states.tolist()[i]) for i in range(3)]
            [cur_states.pop(0) for i in range(3)]
            [delta_pressures.append(feedback.reshape(-1).tolist()[i]) for i in range(3)]
            [delta_pressures.pop(0) for i in range(3)]
            x_in = np.concatenate((np.array(cur_states).reshape(3*m,1),np.array(delta_pressures).reshape(3*n,1)),axis=0)

        iteration_end = iteration
            
        time = np.array([])
        position = np.array([])

        iteration = iteration_begin
        while iteration < iteration_end:
            observation = self.frontend.read(iteration)
            obs_position = np.array( observation.get_positions() )
            obs_time = np.array( observation.get_time_stamp() )*1e-9

            position = np.append(position, obs_position.reshape(-1)[0:3])
            time = np.append(time, obs_time)

            iteration += iterations_per_command
        
        time = time-time[0]
        
        return time, input, position, compute_diff

    def ILC(self, number_iteration, GLOBAL_INITIAL, mode_name='none', ref_traj=None, T_go=None):
        '''
        only get the feedforward control without exciting the simulation/real system
        u is the basic presssure and ff is the feedforward control
        dimensions of u_ago, u_ant and ff are all dofs * length
        '''
        (u_ago, u_ant, ff) = self.Feedforward(self.y_desired)
        ff = np.zeros(ff.shape)
        # avoid too aggressive motion in the first iteration
        # ff = 0.1 * ff
        # only feedforward is used when do ILC
        mode_name_list = ["ff", "ff", "ff", "ff"]
        # generate the initial state z0 and the shifted control input
        # generate the initial disturbance
        disturbance = np.zeros((len(self.dof_list), self.y_desired.shape[1]))
        P_list = []
        for dof in self.dof_list:
            # self.Optimizer_list[dof].GenerateVector( ff[dof, :] )
            '''
            Generate covariance matrices Q, P, R for each dof,
            and record P for the first iteration
            '''
            P = self.Optimizer_list[dof].GenerateCovMatrix(self.delta_d_list[dof], 
                                                           self.delta_y_list[dof],
                                                           self.delta_w_list[dof],
                                                           self.delta_ini_list[dof])
            P_list.append( P )

        # initialization
        y_history = []
        ff_history = []
        disturbance_history = []
        P_history = []
        fb_history = []
        ago_history = []
        ant_history = []
        d_lifted_history = []
        P_lifted_history = []
        repeated = []
        P_lifted = [None] * len(self.dof_list)
        d_lifted = np.zeros(disturbance.shape)
        # disturbance_history.append( np.copy( disturbance ) )
        # P_history.append( np.copy( P_list ) )
        y_history.append( np.copy( self.y_desired ) )

        # begin to learn
        for i in range(number_iteration):
            
            ff_history.append( np.copy( ff ) )
            print("Iteration: {}".format(i))

            '''
            read the output of the simulation/real system
            y is the absolute measured angle
            ''' 
            print("Begin to measure...")
            (y, fb, obs_ago, obs_ant, des_ago, des_ant, _) = self.Control(self.y_desired, mode_name_list=mode_name_list, 
                                                     ifplot="no", u_ago=u_ago, u_ant=u_ant, 
                                                     ff=ff, echo="yes",
                                                     controller='pd' )
            print("...measurement completed")

            '''
            calculate the variables for the kalman filter
            ff + u is used to reach the absolute angle
            P and disturbance will be updated using Kalman filter
            '''
            print("Begin to optimize...")
            for dof in self.dof_list:
                (ff_temp, dis_temp, P_temp, dis_lifted_temp, P_lifted_temp) \
                = self.Optimizer_list[dof].Optimization(y[dof, :], 
                                                        ff[dof, :], 
                                                        disturbance[dof, :], 
                                                        P_list[dof], 
                                                        number_iteration=i, 
                                                        weight=self.weight_list[dof],
                                                        mode_name=mode_name )
                
                d_lifted[dof, :] = np.copy( dis_lifted_temp.reshape(1, -1) )
                P_lifted[dof] = np.copy(P_lifted_temp )

                ff[dof, :] = np.copy( ff_temp.reshape(1, -1) )
                disturbance[dof, :] = np.copy( dis_temp.reshape(1, -1) )
                P_list[dof] = np.copy( P_temp )
            print("...optimization completed")

            t_stamp = np.linspace(0, (y.shape[1]-1)*0.01, y.shape[1], endpoint = True)
            y_ = y - y[:,0].reshape(-1,1) + ref_traj[:,0].reshape(-1,1)

            RG = RobotGeometry(initial_posture=PAMY_CONFIG.GLOBAL_INITIAL)
            (_, end_ref) = RG.AngleToEnd(ref_traj)
            (_, end_real) = RG.AngleToEnd(y)
            wandb_plot(i_iter=i, frequency=1, t_stamp=t_stamp, ff=ff, fb=fb, y=y_, theta_=ref_traj, t_stamp_list=[], theta_list=[], T_go_list=[T_go], p_int_record=[], 
                   obs_ago=obs_ago, obs_ant=obs_ant, des_ago=des_ago, des_ant=des_ant, SI_ref = None, disturbance=disturbance, end_ref=end_ref, end_real=end_real)

            # record all the results of each iteration
            fb_history.append(np.copy(fb))
            ago_history.append(np.copy(obs_ago))
            ant_history.append(np.copy(obs_ant))
            d_lifted_history.append(np.copy( d_lifted ))
            P_lifted_history.append(np.copy( P_lifted ))
            disturbance_history.append(np.copy( disturbance))
            y_history.append(np.copy(y))
            P_history.append(np.copy(P_list))

            # set the same initial angle for the next iteration
            print("Begin to initialize...")
            # self.AngleInitialization(GLOBAL_INITIAL)
            (ig_t, ig_step, ig_position, ig_diff, ig_theta_zero) = self.LQRTesting(amp = np.array([[30], [30], [30]])/180*math.pi, t_start = 0.0, t_duration = 6.0)
            self.PressureInitialization()
            print("...initialization completed")

        # for _ in range(2):
        #     (y, _, _, _) = self.Control(self.y_desired, mode_name_list=mode_name_list, 
        #                                 ifplot="no", u_ago=u_ago, u_ant=u_ant, ff=ff, echo="yes", 
        #                                 controller='pd' )
            
        #     repeated.append(np.copy(y))
        #     # self.AngleInitialization(GLOBAL_INITIAL)
        #     (ig_t, ig_step, ig_position, ig_diff, ig_theta_zero) = self.LQRTesting(amp = np.array([[30], [30], [30]])/180*math.pi, t_start = 0.0, t_duration = 6.0)
        #     self.PressureInitialization()

        # mode_name_list = ['ff+fb', 'ff+fb', 'ff+fb', 'ff+fb']
        # (y_pid, _, _, _) = self.Control(self.y_desired, mode_name_list=mode_name_list, 
        #                                 ifplot="no", u_ago=u_ago, u_ant=u_ant, ff=ff, echo="yes", 
        #                                 controller='pd' )
        # # self.AngleInitialization(GLOBAL_INITIAL)
        # (ig_t, ig_step, ig_position, ig_diff, ig_theta_zero) = self.LQRTesting(amp = np.array([[30], [30], [30]])/180*math.pi, t_start = 0.0, t_duration = 6.0)
        # self.PressureInitialization()

        y_pid = None
        return(y_history, repeated, ff_history, disturbance_history, P_history, d_lifted_history, P_lifted_history, fb_history, ago_history, ant_history, y_pid)

    def get_delay(self, dof, xx):
        delay = self.ndelay_list[dof]
        xx_ = np.zeros(len(xx))
        xx_[0:len(xx)-delay] = xx[delay:]
        xx_[len(xx)-delay:] = xx[-1]
        return xx_

    def online_convex_optimization(self, b_list, mode_name='ff', coupling='no', learning_mode='b', trainable_fb=None, device=None, dim_fb=None):
        mode_name_list = [mode_name, mode_name, mode_name, mode_name]
        ff    = np.zeros(self.y_desired.shape) 
        y_des = np.zeros(self.y_desired.shape)
        u_ago = np.ones(self.y_desired.shape) * self.anchor_ago_list.reshape(-1, 1)
        u_ant = np.ones(self.y_desired.shape) * self.anchor_ant_list.reshape(-1, 1)
        
        # if mode_name == 'ff':
        for i in range(len(b_list)):
            if learning_mode == 'u':
                ff[i, :] = b_list[i].flatten()
            elif learning_mode == 'b':
                if coupling == 'yes':
                    ff[i, :] = (self.Xi@b_list[i]).flatten()
                else:
                    ff_ = (self.O_lsit[i].Xi@b_list[i]).flatten()
                    # ff_ = self.get_delay(i, ff_)
                    ff[i, :] = ff_
        (y, fb, obs_ago, obs_ant, des_ago, des_ant, fb_datasets) = self.Control(self.y_desired, mode_name_list=mode_name_list, ifplot="no", u_ago=u_ago, u_ant=u_ant, 
                                                    ff=ff, echo="yes", controller='pd', trainable_fb=trainable_fb, device=device, dim_fb=dim_fb)
        u_ff = np.copy(ff)
            
        # elif mode_name == 'ff+fb' or mode_name == 'fb+ff':
        #     for i in range(len(b_list)):
        #         y_des[i, :] = (self.O_lsit[i].Xi@b_list[i]).flatten()

        #     (y, fb, obs_ago, obs_ant) = self.Control(y_des, mode_name_list=mode_name_list, ifplot="no", u_ago=u_ago, u_ant=u_ant, 
        #                                              ff=ff, echo="yes", controller='pd')
        
        #     u_ff = np.copy(y_des)

        return (y, u_ff, fb, obs_ago, obs_ant, des_ago, des_ant, fb_datasets)
