import math
import numpy as np
import time
import matplotlib.pyplot as plt
import o80
from scipy import integrate
# %% Build the model for a single joint
class Joint:
    def __init__(self, frontend, dof, anchor_ago, 
                 anchor_ant, num=None, den=None, order_num=None, 
                 order_den=None, ndelay=None, pid=None, 
                 f_frontend=100, f_backend=500):

        self.dof        = dof
        self.frontend   = frontend
        self.anchor_ago = anchor_ago
        self.anchor_ant = anchor_ant
        self.delay      = ndelay
        self.order_num  = order_num
        self.order_den  = order_den
        if num is None:
            pass
        else:
            self.num    = num[0:order_num+1]
        if den is None:
            pass
        else:
            self.den    = den[0:order_den+1]
        self.pid        = pid
        self.f_frontend = f_frontend
        self.f_backend  = f_backend
        self.p_frontend = 1.0/self.f_frontend # period of frontend
        self.p_backend  = 1.0/self.f_backend  # period of backend
        self.dt         = 1.0/self.f_frontend
        self.ipc        = int( self.p_frontend/self.p_backend ) # ipc: iteration per command

    def Feedforward(self, y):
        '''
        calculate the forward input
        here y should be the relative angles
        '''
        u_ago = []
        u_ant = []
        ff = []
        for i in range(0, len( y )):

            sum_num = 0
            for Nr in range(self.order_num+1):
                a = i + self.delay - Nr
                if a >= len(y):
                    a = len(y) - 1
                if a >= 0:
                    term = self.num[Nr] * (y[a] - y[i])
                else:
                    term = self.num[Nr] * 0.0
                sum_num += term
            
            sum_den = 0
            for Nr in range(1, self.order_den+1):
                a = i - Nr
                if a >= 0:
                    term = self.den[Nr] * ff[a]
                else:
                    term = self.den[Nr] * 0.0
                sum_den += term

            feedforward = sum_num - sum_den 
            ff.append(feedforward)
            u_ago.append(self.anchor_ago)
            u_ant.append(self.anchor_ant)

        u_ago = np.array(u_ago)
        u_ant = np.array(u_ant)
        ff = np.array(ff)

        return(u_ago, u_ant, ff)

    def PlotFigures(self, y, angle_initial, ff, fb, t_stamp_u, iteration_begin, iteration_end, iterations_per_command):
        
        y = np.array(y)

        iteration = iteration_begin
        obs_pressure_ago = []
        obs_pressure_ant = []
        des_pressure_ago = []
        des_pressure_ant = []
        position = []
        velocity = []
        t_stamp = []

        while iteration < iteration_end:
            observation = self.frontend.read(iteration)

            obs_pressure = observation.get_observed_pressures()
            des_pressure = observation.get_desired_pressures()
            obs_position = observation.get_positions()
            obs_velocity = observation.get_velocities()

            obs_pressure_ant.append(obs_pressure[self.dof][0])
            obs_pressure_ago.append(obs_pressure[self.dof][1])

            des_pressure_ant.append(des_pressure[self.dof][0])
            des_pressure_ago.append(des_pressure[self.dof][1])

            position.append(obs_position[self.dof])
            velocity.append(obs_velocity[self.dof])
    
            t_stamp.append( observation.get_time_stamp() * 1e-9 )

            iteration += iterations_per_command

        initial_time = t_stamp[0]
        t_stamp = np.array(t_stamp) - initial_time

        print("begin to draw the plot")
        plt.figure(1)
        line_1, = plt.plot(t_stamp, position, label=r'Observed angle $\theta_{obs}$', linewidth=1)
        line_2, = plt.plot(t_stamp_u, y + angle_initial, label=r'Desired angle $\theta_{des}$', linewidth=0.3)
        #plt.legend(handles = [line_1, line_2], loc='upper right', shadow=True)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Angle $\theta$ in rad')
        plt.show()

        # plt.figure(2)
        # plt.plot(t_stamp, ff, linewidth=1)
        # # plt.legend(handles = [line_1, line_2], loc='upper right', shadow=True)
        # plt.xlabel(r'Time $t$ in s')
        # plt.ylabel(r'Feedforward Control')
        # plt.show()

        plt.figure(3)
        plt.plot(t_stamp, fb, linewidth=1)
        # plt.legend(handles = [line_1, line_2], loc='upper right', shadow=True)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Feedback Control')
        plt.show()

    def Control(self, y=[], mode_name="fb+ff", mode_trajectory="ref",
                ifplot="yes", u_ago=[], u_ant=[], ff=[], echo="no"):

        if len(y) != 0:
            t_stamp_u = np.linspace(0, (len(y) - 1)/self.f_frontend, len(y) )
        elif len(ff) != 0:
            t_stamp_u = np.linspace(0, (len(ff) - 1)/self.f_frontend, len(ff) )

        theta           = self.frontend.latest().get_positions()
        pressures       = self.frontend.latest().get_observed_pressures()
        pressure_ago    = pressures[self.dof][0]
        pressure_ant    = pressures[self.dof][1]

        if mode_trajectory == "ref":
            angle_initial = theta[self.dof]
            print("Initial angle: {:.2f} degree".format(angle_initial * 180 / math.pi))
        elif mode_trajectory == "abs":
            angle_initial = 0.0

        if mode_name != 'fb':
            if (len(u_ago) == 0) or (len(ff) == 0) or (len(u_ant)==0):
                (u_ago, u_ant, ff) = self.Feedforward(y)

        fb    = np.array([])
        fb_kp = np.array([])
        fb_ki = np.array([])
        fb_kd = np.array([])
        angle_delta_list = np.array([])
        res_i_list = np.array([])
        res_d_list = np.array([])

        iteration_reference = self.frontend.latest().get_iteration()  # read the current iteration
        iteration_begin     = iteration_reference + 1000  # set the iteration when beginning
        iteration           = iteration_begin

        self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                  o80.Iteration(iteration_begin-self.ipc),
                                  o80.Mode.QUEUE)
        self.frontend.pulse()

        res_i = 0
        angle_delta_pre = 0

        for i in range(0, len( t_stamp_u )):
            
            if mode_name != 'ff':
                angle_delta = y[i] + angle_initial - theta[self.dof]
                res_d       = (angle_delta - angle_delta_pre)/self.dt
                res_i      += angle_delta * self.dt
                angle_delta_list = np.append(angle_delta_list, angle_delta)
                res_i_list = np.append(res_i_list, res_i)
                res_d_list = np.append(res_d_list, res_d)
                feedback_kp = self.pid[0] * angle_delta
                feedback_ki = self.pid[1] * res_i
                feedback_kd = self.pid[2] * res_d
                feedback    = self.pid[0] * angle_delta + self.pid[1] * res_i + self.pid[2] * res_d
                fb          = np.append(fb, feedback)
                fb_kp       = np.append(fb_kp, feedback_kp)
                fb_ki       = np.append(fb_ki, feedback_ki)
                fb_kd       = np.append(fb_kd, feedback_kd)
                angle_delta_pre = angle_delta

            if mode_name == "ff":
                diff = int(ff[i]) 
            elif mode_name == "fb":
                diff = int(fb[i])
            elif mode_name == "fb+ff" or mode_name == "ff+fb":
                diff = int(ff[i] + fb[i])
            
            pressure_ago = int(self.anchor_ago + diff)
            pressure_ant = int(self.anchor_ant - diff)

            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
            self.frontend.pulse()
            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + self.ipc - 1),
                                      o80.Mode.QUEUE)

            observation = self.frontend.pulse_and_wait()
            theta = observation.get_positions()
            iteration += self.ipc
        
        iteration_end = iteration
        
        if ifplot == "yes":
            self.PlotFigures(y, angle_initial, ff, fb, t_stamp_u, iteration_begin, iteration_end, self.ipc)
        
        if echo == "yes":
            position = []
            iteration = iteration_begin
            pressure_ago = np.array([])
            pressure_ant = np.array([])
            
            while iteration < iteration_end:
                observation = self.frontend.read(iteration)
                obs_position = np.array(observation.get_positions())
                obs_pressure = np.array(observation.get_observed_pressures())
                pressure_ago = np.append(pressure_ago, obs_pressure[self.dof][0])
                pressure_ant = np.append(pressure_ant, obs_pressure[self.dof][1])
                position.append(obs_position[self.dof])
                iteration += self.ipc

            return(position, fb, pressure_ago, pressure_ant, fb_kp, fb_ki, fb_kd, angle_delta_list, res_i_list, res_d_list)
    

    def PressureInitialization(self, pressure_ago, pressure_ant, times=1, duration=2):
        for _ in range(times):
            # creating a command locally. The command is *not* sent to the robot yet.
            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Duration_us.seconds(duration),
                                      o80.Mode.QUEUE)

            # sending the command to the robot, and waiting for its completion.
            self.frontend.pulse_and_wait()

        observation = self.frontend.latest()
        pressures = observation.get_observed_pressures()
        positions = observation.get_positions()

        print("{} dof is: {}".format(self.dof, pressures[self.dof]))

    def AngleInitialization(self, angle, tolerance=0.1, T=5):
        theta = self.frontend.latest().get_positions()
        theta_dot = self.frontend.latest().get_velocities()
        iteration = self.frontend.latest().get_iteration() + 200  # set the iteration when beginning
        self.frontend.add_command(self.dof, self.anchor_ago, self.anchor_ant,
                                o80.Iteration(iteration-self.ipc),
                                o80.Mode.QUEUE)
        
        self.frontend.pulse()

        res_i = 0
        angle_delta_pre = 0

        while (abs((theta[self.dof] - angle)*180/math.pi) > tolerance) or (abs(theta_dot[self.dof] - 0.00) > 0.001):
            angle_delta = angle - theta[self.dof]
            res_d = ( angle_delta - angle_delta_pre ) / self.dt
            res_i += angle_delta * self.dt
            feedback = self.pid[0] * angle_delta + self.pid[1] * res_i + self.pid[2] * res_d
            angle_delta_pre = angle_delta

            diff = int( feedback )

            pressure_ago = int( self.anchor_ago + diff )
            pressure_ant = int( self.anchor_ant - diff )

            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
            self.frontend.pulse()
            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + self.ipc - 1),
                                      o80.Mode.QUEUE)

            observation = self.frontend.pulse_and_wait()
            theta = observation.get_positions()
            theta_dot = observation.get_velocities()
            iteration += self.ipc

        print("Expected: {:.2f} degree".format(angle * 180 / math.pi))
        print("Actual: {:.2f} degree".format(theta[self.dof] * 180 / math.pi) )
        print("Error: {:.2f} degree".format( ( theta[self.dof] - angle ) * 180 / math.pi ))