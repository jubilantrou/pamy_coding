import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MinJerk_penalty as MJ_penalty
import MinJerk_analytic as MJ_analytic
import LinearPlanning as MJ_linear
import numba as nb
import pickle5 as pickle
import random

class RobotGeometry:

    def __init__(self, center_of_robot=[0, 0, 0], x_of_robot=[1, 0, 0],
                 y_of_robot=[0, 1, 0], z_of_robot=[0, 0, 1], 
                 initial_posture=np.array([0, 0.833, 0.220, 0]), #initial_posture=np.array([0, -30, -30, 0])/180*math.pi, 
                 l_1=0.40, l_2=0.38, step_size=0.01, constraint_angle=None):
        # for now the coordinates of robot must be [0, 0, 0]
        self.center_of_robot = center_of_robot
        self.x_of_robot = x_of_robot
        self.y_of_robot = y_of_robot
        self.z_of_robot = z_of_robot
        # initial posture is given in joint space
        self.initial_posture = initial_posture
        self.l_1 = l_1
        self.l_2 = l_2
        self.step_size = step_size
        self.constraint_angle = constraint_angle
        (_, self.position_final) = self.AngleToEnd(self.initial_posture[0:3],  frame='Cylinder')
        self.acceleration_final = [0, 0, 0]

    # def UpdateInitial(angle):
    #     self.initial_posture = angle
    #     (_, self.position_final) = self.AngleToEnd(self.initial_posture[0:3],  frame='Cylinder')

    # from angle space to end effector space in Cartesian Coordinate or Polar Coordinate
    def AngleToEnd(self, angle, frame='Cartesian'):
        # from joint space to end effector space
        # position_A is the end effector of the first robot arm
        # position_B is the end effector of the second robot arm
        # can be transformed into Cartesian space, Polar space or Cylinder space
        l_1 = self.l_1
        l_2 = self.l_2

        theta0 = angle[0]
        theta1 = angle[1]
        theta2 = angle[2]

        if frame == 'Cartesian':
            x = math.cos(theta0) * math.sin(theta1) * l_1
            y = math.sin(theta0) * math.sin(theta1) * l_1
            z = math.cos(theta1) * l_1
            position_A = np.array([-y, x, z+1.21])
            x = math.cos(theta0) * math.sin(theta1) * l_1 + math.cos(theta0) * math.sin(theta1 + theta2) * l_2
            y = math.sin(theta0) * math.sin(theta1) * l_1 + math.sin(theta0) * math.sin(theta1 + theta2) * l_2
            z = math.cos(theta1) * l_1 + math.cos(theta1 + theta2) * l_2
            position_B = np.array([-y, x, z+1.21])

        elif frame == 'Polar':
            position_A = np.array([theta0, theta1, l_1])
            r = np.sqrt( l_1**2 + l_2**2 - 2*l_1*l_2*math.cos(math.pi-abs(theta2)) )
            gama = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
            if theta2 > 0:
                alpha = theta1 + gama
            else:
                alpha = theta1 - gama
            position_B = np.array([theta0, alpha, r])

        elif frame == 'Cylinder':
            # theta0, r, z
            position_A = np.array([theta0, abs(l_1*math.sin(theta1)), (l_1*math.cos(theta1))+1.21])

            r = np.sqrt( l_1**2 + l_2**2 - 2*l_1*l_2*math.cos(math.pi-abs(theta2)) )

            gama = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
            if theta2 > 0:
                alpha = theta1 + gama
            else:
                alpha = theta1 - gama
            position_B = np.array([theta0, abs(r*math.sin(alpha)), (r*math.cos(alpha))+1.21])

        return (position_A, position_B)

    def EndToAngle( self, position, frame='Cartesian'):
        l_1 = self.l_1
        l_2 = self.l_2
        # transform coordinates in Cartesian space into joint space
        if frame == 'Cartesian':
            x = position[1]
            y = -position[0]
            z = position[2]-1.21

            l      = np.linalg.norm((x,y,z), ord=2)
            theta0 = math.atan(y/x)

            temp = (l_1**2 + l**2 - l_2**2) / (2*l_1*l)
            if temp>1:
                temp=1
            if temp<-1:
                temp=-1
            gamma  = math.acos(temp)
            alpha  = math.asin(z/l)
            theta1 = math.pi/2 - gamma - alpha

            temp1 = (l_1**2 + l_2**2 - l**2) / (2*l_1*l_2)
            if temp1>1:
                temp1=1
            if temp1<-1:
                temp1=-1
            beta   = math.acos(temp1)
            theta2 = math.pi - beta

            angle  = np.array( [theta0, theta1, theta2] )

        return angle

    def CalAngularTrajectory(self, trajectory, angle, frame='Cartesian'):
        # transform trajectory in end effector space into joint space
        l_1 = self.l_1
        l_2 = self.l_2

        eps = 1e-10
        angle_trajectory = np.array([])

        if frame == 'Cartesian':
            for i in range(trajectory.shape[1]):
                x = trajectory[0, i]
                y = trajectory[1, i]
                z = trajectory[2, i]

                theta0 = angle[0]
                theta1 = angle[1]
                theta2 = angle[2]

                l = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                if l > l_1 + l_2:
                    x = (l_1 + l_2) / l * x
                    y = (l_1 + l_2) / l * y
                    z = (l_1 + l_2) / l * z

                cos_theta0 = x / np.sqrt((x ** 2 + y ** 2))
                sin_theta0 = y / np.sqrt((x ** 2 + y ** 2))

                theta0_ = math.acos( cos_theta0 )
                if sin_theta0 < 0:
                    theta0_ = - abs(theta0_)
                
                if cos_theta0 != 0:
                    x_tilde = x / cos_theta0
                else:
                    x_tilde = y / sin_theta0
                
                l_tilde = np.sqrt( z**2 + x_tilde**2 )

                theta2_temp = math.acos( - round((l_tilde ** 2 - l_1 ** 2 - l_2 ** 2) / (2 * l_1 * l_2), 4) )
                theta2_ = math.pi - theta2_temp
                if abs(-theta2_ - theta2) < abs(theta2_ - theta2):
                    theta2_ = -theta2_

                theta1_temp1 = math.acos( -round((l_2 ** 2 - l_1 ** 2 - l_tilde ** 2) / (2 * l_1 * l_tilde), 4) )
                theta1_temp2 = math.asin( z / l_tilde )
                if theta2_ < 0:
                    theta1_ = math.pi / 2 - theta1_temp2
                else:
                    theta1_ = math.pi / 2 - theta1_temp1 - theta1_temp2

                angle = np.array([theta0_, theta1_, theta2_])

                angle_trajectory = np.append(angle_trajectory, angle)
        
        elif frame == 'Polar':
            for i in range(trajectory.shape[1]):
                theta0 = trajectory[0, i]
                alpha = trajectory[1, i]
                r = trajectory[2, i]

                theta1 = angle[1]
                theta2 = angle[2]

                theta0_ = theta0

                theta2_ = math.pi - math.acos( (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) )

                if abs(-theta2_ - theta2) < abs(theta2_ - theta2):
                    theta2_ = -theta2_

                theta_temp = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
                if theta2_ > 0:
                    theta1_ = alpha - theta_temp
                else:
                    theta1_ = alpha + theta_temp

                angle = np.array([theta0_, theta1_, theta2_])

                angle_trajectory = np.append(angle_trajectory, angle)
        
        elif frame == 'Cylinder':
            # dimension of trajectory: 3 * length
            angle_trajectory = np.zeros(trajectory.shape)

            for i in range(trajectory.shape[1]):

                theta0 = trajectory[0, i]
                l = trajectory[1, i]
                z = trajectory[2, i]-1.21

                r = np.sqrt( l**2 + z**2 ) if np.sqrt( l**2 + z**2 ) <= l_1+l_2 else l_1+l_2

                if abs(z/r) <= 1:
                    z_over_r = z/r
                elif z/r < 0:
                    z_over_r = -1
                elif z/r > 0:
                    z_over_r = 1
                # alpha = -abs( math.pi/2 - math.asin( z_over_r) )
                alpha = abs( math.pi/2 - math.asin( z_over_r) )
                 
                theta2 = angle[2]
                theta0_ = theta0

                if abs ((l_1**2 + l_2**2 - r**2) / (2*l_1*l_2)) <= 1:
                    l_l = (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2)
                elif (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) > 0:
                    l_l = 1
                elif (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) < 0:
                    l_l = -1
                theta2_ = math.pi - math.acos( l_l )

                if abs(-theta2_ - theta2) < abs(theta2_ - theta2):
                    theta2_ = -theta2_

                theta_temp = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
                if theta2_ > 0:
                    theta1_ = alpha - theta_temp  
                else:
                     theta1_ = alpha + theta_temp
    
                angle_trajectory[:, i] = np.array([theta0_, theta1_, theta2_])     

        return angle_trajectory

    # def TransInPolar(self, position, velocity):
    #     theta0 = position[0]
    #     alpha = position[1]
    #     r = position[2]

    #     # rotate around z-axis
    #     R_1 = np.array([[math.cos(theta0), -math.sin(theta0), 0],
    #                     [math.sin(theta0), math.cos(theta0), 0],
    #                     [0, 0, 1]])
    #     # rotate around x-axis
    #     R_2 = np.array([[1, 0, 0],
    #                     [0, math.cos(- alpha), -math.sin(- alpha)],
    #                     [0, math.sin(- alpha), math.cos(- alpha)]])
    #     # total rotation
    #     R = np.dot(R_2, R_1)

    #     ex = np.array([1, 0, 0]).reshape(-1, 1)
    #     ey = np.array([0, 1, 0]).reshape(-1, 1)
    #     ez = np.array([0, 0, 1]).reshape(-1, 1)

    #     e_r = np.dot(R, ex)
    #     e_theta0 = np.dot(R, ey)
    #     e_alpha = np.dot(R, ez)

    #     v_r = np.asscalar( np.dot(velocity, e_r) )
    #     v_theta0 = np.asscalar( np.dot(velocity, e_theta0) )
    #     v_alpha = np.asscalar( np.dot(velocity, e_alpha) )

    #     velocity_polar = np.array([v_theta0, v_alpha, v_r])

    #     return( velocity_polar )

    # def RotationMatrix(self, angle):
    #     x = angle[0]
    #     y = angle[1]
    #     z = angle[2]
    #     R_x = np.array([[1, 0, 0],
    #                     [0, math.cos(x), -math.sin(x)],
    #                     [0, math.sin(x), math.cos(x)]])

    #     R_y = np.array([[math.cos(y), 0, math.sin(y)],
    #                     [0, 1, 0],
    #                     [-math.sin(y), 0, math.cos(y)]])
        
    #     R_z = np.array([[math.cos(z), -math.sin(z), 0],
    #                     [math.sin(z), math.cos(z), 0],
    #                     [0, 0, 1]])
        
    #     R = np.dot(R_y, R_x)
    #     R = np.dot(R_z, R)
    #     return R
    
    # def CheckConstraint(self, position ):
    #     flag = True

    #     if np.linalg.norm(position, ord=2) >= self.l_1 + self.l_2:
    #         flag = False
    #     else:
    #         angle = self.EndToAngle( position )
    #         for dof in range(3):
    #             if angle[dof]>=self.constraint_angle[dof,0] and angle[dof]<=self.constraint_angle[dof,1]:
    #                 pass
    #             else:
    #                 flag = False
    #                 break
    #     return flag

    def updatedPathPlanning(self, time_point, T_go=1.0, T_back=1.0, T_steady=0.1,
                      angle=None, velocity_initial=np.array([0, 0, 0]), 
                      acceleration_initial=np.array([0, 0, 0]),
                      target=None, frequency=100, plan_weight=(6, 10)):
        
        p_list = []
        v_list = []
        a_list = []
        j_list = []
        theta_list = []
        t_stamp_list = []
        p_int_record = [target]
        time_update_record = []

        (p, v, a, j, theta, t_stamp) = self.PathPlanning(time_point=time_point, angle=angle, T_go=T_go, target=target)
        print(theta[:,-1]/math.pi*180)
        p_list.append(p)
        v_list.append(v)
        a_list.append(a)
        j_list.append(j)
        theta_list.append(theta)
        t_stamp_list.append(t_stamp)

        idx_begin = 30
        t_begin = 0.3
        while t_stamp[-1]-t_begin>=0.4:
            print('currrent time:')
            print(t_begin)
            dice = random.random()
            if dice>0.5:
                print('no change')
                idx_begin += 10
                t_begin += 0.1
                continue
            else:
                T_go += random.randrange(-15, 15)/100
                (_, temp) = self.AngleToEnd(target[0:3], frame='Cartesian')
                temp += [random.randrange(-4,4)/100,random.randrange(-4,4)/100,random.randrange(-8,8)/100]
                target = self.EndToAngle(temp)
                p_int_record.append(target)
                print('update')
                print('pos: {}'.format(theta[:,(idx_begin+10)]))
                print('vel: {}'.format(v[:,(idx_begin+10)]))
                print('acc: {}'.format(a[:,(idx_begin+10)]))
                print('T_go: {}'.format(T_go))
                print('target: {}'.format(target))
                if T_go-(t_begin+0.1) >= 0.4:
                    (p, v, a, j, theta, t_stamp) = self.PathPlanning(time_point=(t_begin*100+10), angle=theta[:,(idx_begin+10)], velocity_initial=v[:,(idx_begin+10)], acceleration_initial=a[:,(idx_begin+10)], T_go=T_go, target=target)
                else:
                    (p, v, a, j, theta, t_stamp) = self.PathPlanning(time_point=(t_begin*100+10), angle=theta[:,(idx_begin+10)], velocity_initial=v[:,(idx_begin+10)], acceleration_initial=a[:,(idx_begin+10)], T_go=T_go, target=target, part=0)
                p_list.append(p)
                v_list.append(v)
                a_list.append(a)
                j_list.append(j)
                theta_list.append(theta)
                t_stamp_list.append(t_stamp)
                time_update_record.append(idx_begin+10)
                idx_begin = 0
                t_begin += 0.1

        
        p_final = np.hstack([p_list[i][:,:time_update_record[i]] for i in range(len(time_update_record))]+[p_list[-1]])
        v_final = np.hstack([v_list[i][:,:time_update_record[i]] for i in range(len(time_update_record))]+[v_list[-1]])
        a_final = np.hstack([a_list[i][:,:time_update_record[i]] for i in range(len(time_update_record))]+[a_list[-1]])
        j_final = np.hstack([j_list[i][:,:time_update_record[i]] for i in range(len(time_update_record))]+[j_list[-1]])
        theta_final = np.hstack([theta_list[i][:,:time_update_record[i]] for i in range(len(time_update_record))]+[theta_list[-1]])
        t_stamp_final = np.hstack([t_stamp_list[i][:time_update_record[i]] for i in range(len(time_update_record))]+[t_stamp_list[-1]])
        
        if len(time_update_record)==0:
            print('output no change')
            return (p, v, a, j, theta, t_stamp, theta_list, t_stamp_list, p_int_record, time_update_record)
        return (p_final, v_final, a_final, j_final, theta_final, t_stamp_final, theta_list, t_stamp_list, p_int_record, time_update_record)

    def PathPlanning( self, time_point, T_go=1.0, T_back=1.0, T_steady=0.1,
                      angle=None, velocity_initial=np.array([0, 0, 0]), 
                      acceleration_initial=np.array([0, 0, 0]),
                      target=None, frequency=100, plan_weight=(6, 10), part=1,):

        l_1 = self.l_1
        l_2 = self.l_2
        pi = math.pi
        '''
        angle is absolute variable in angular space
        target is absolute variable in angular space
        '''
        (_, p_initial) = self.AngleToEnd(angle[0:3], frame='Cylinder')  # theta1, r, h
        (_, p_target) = self.AngleToEnd(target[0:3], frame='Cylinder')  # theta1, r, h

        v_target = np.array([4.0/p_target[1], 0, 0])
        v_final = np.array([0, 0, 0])
        a_target = np.array([0, 0, 0])
        a_final = np.array([0, 0, 0])
        
        if time_point >= round(T_go * frequency):
            m_list = [[1.0, 1.0],
                      [1.0, 1.0],
                       [1.0, 1.0]]
            m_list = np.array(m_list)
            
            n_list = [[4.0, 0.1],
                      [1.0, 0.1],
                      [1.0, 0.1]]
            n_list = np.array(n_list)

            # target is given in joint space
            t = np.array([time_point/frequency, T_go+T_back, T_go+T_back+T_steady])
            # corresponding positions
            p = np.array([p_initial, self.position_final, self.position_final]).T
            p = p - p_initial.reshape(-1, 1)
            # corresponding velocities
            v = np.array([velocity_initial, v_final, v_final]).T
            a = np.array([acceleration_initial, a_final, a_final]).T

        elif time_point < round(T_go * frequency):
            m_list = [[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]]
            m_list = np.array(m_list)
            n_list = [[plan_weight[0], plan_weight[1], 0.1],
                      [1.0, 1.0, 0.1],
                      [1.0, 1.0, 0.1]]
            n_list = np.array(n_list)
            t = np.array([time_point/frequency, T_go, T_go+T_back, T_go+T_back+T_steady])

            # angle_list = np.array([ angle[0:3], 
            #                         target[0:3], 
            #                         self.initial_posture[0:3], 
            #                         self.initial_posture[0:3] ]).T
            # angle_list = angle_list - angle[0:3].reshape(-1, 1) - self.initial_posture[0:3].reshape(-1, 1)
            # corresponding positions
            p = np.array([p_initial, p_target, self.position_final, self.position_final]).T
            p = p - p_initial.reshape(-1, 1)
            # corresponding velocities
            v = np.array([velocity_initial, v_target, v_final, v_final]).T
            a = np.array([acceleration_initial, a_target, a_final, a_final]).T
        
        # [p_angular_mjl, t_stamp] = MJ_linear.PathPlanning(angle_list, v, t, 1/frequency)
        if part:
            [p_mjp, p_mjv, p_mja, p_mjj, t_stamp] = MJ_penalty.PathPlanning(p[:,:2], v[:,:2], a[:,:2], t[:2], 1/frequency, m_list, n_list)
        else:
            [p_mjp, p_mjv, p_mja, p_mjj, t_stamp] = MJ_penalty.PathPlanning(p, v, a, t, 1/frequency, m_list, n_list)    
        p_mjp = p_mjp + p_initial.reshape(-1, 1)
        # start = time.perf_counter()
        # [p_mja, _, _, _] = MJ_analytic.PathPlanning(1/frequency, t, p, v, a, smooth_acc=True)
        # p_mja = p_mja + p_initial.reshape(-1, 1)

        # p_angular_mjp = self.CalAngularTrajectory(p_mjp, angle[0:3] + self.initial_posture[0:3], frame='Cylinder')
        p_angular_mjp = self.CalAngularTrajectory(p_mjp, self.initial_posture[0:3], frame='Cylinder')
        # p_angular_mja = self.CalAngularTrajectory(p_mja, angle[0:3] + self.initial_posture[0:3], frame='Cylinder')

        return (p_mjp, p_mjv, p_mja, p_mjj, p_angular_mjp, t_stamp)

def GetPlot(p_mja, p_mjp, p_angular_mja, p_angular_mjp, p_angular_mjl, t_list, step, index):

    nr = t_list / step
    nr = nr.astype(np.int)
    nr_dof = p_mja.shape[0]
    nr_point = p_mja.shape[1]
    t_stamp = np.linspace(0, nr_point*step, nr_point, endpoint=True)

    legend_position = 'lower right'
    fig, axs = plt.subplots(2, nr_dof, figsize=(16, 8))
    
    for i_dof in range(nr_dof):
        
        ax = axs[0, i_dof]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Angle $\theta$ in degree')
        line = []
        line_temp, = ax.plot(t_stamp, p_angular_mjp[i_dof, :] * 180 / math.pi, linewidth=1.5, label=r'Penalty')
        line.append( line_temp )
        line_temp, = ax.plot(t_stamp, p_angular_mja[i_dof, :] * 180 / math.pi, linewidth=1.5, label=r'Analytic')
        line.append( line_temp )
        line_temp, = ax.plot(t_stamp, p_angular_mjl[i_dof, :] * 180 / math.pi, linewidth=1.5, label=r'Linear')
        line.append( line_temp )
        for i in range(len(t_list)):
            ax.axvline(t_stamp[nr[i]], color='red', linewidth=0.5, linestyle='--')
        ax.legend(handles=line, loc=legend_position, shadow=True)

        ax = axs[1, i_dof]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'In Polar system')
        line = []
        line_temp, = ax.plot(t_stamp, p_mjp[i_dof, :], linewidth=1.5, label=r'Penalty')
        line.append( line_temp )
        line_temp, = ax.plot(t_stamp, p_mja[i_dof, :], linewidth=1.5, label=r'Analytic')
        line.append( line_temp )
        for i in range(len(t_list)):
            ax.axvline(t_stamp[nr[i]], color='red', linewidth=0.5, linestyle='--')
        ax.legend(handles=line, loc=legend_position, shadow=True)

    # plt.show() 
    plt.savefig("/home/hao/Desktop/HitBall/MinJerkImage/" + str(index) + '.pdf')



if __name__ == '__main__':
    # Robot = RobotGeometry()
    
    # # %% read data of balls
    # path_of_file = "/home/hao/Desktop/Hao/" + 'BallsData' + '.txt'
    # file = open(path_of_file, 'rb')
    # time_list = pickle.load(file)
    # position_list = pickle.load(file)
    # velocity_list = pickle.load(file)
    # file.close()

    # position_mean = np.mean( position_list, axis=0 )
    # # offset angles of the upright posture
    # offset = np.array( [2.94397627, -0.078539855235, -0.06333859293225] )
    # # angles of initial posture
    # angle_initial_ref = np.array( [2.94397627, -0.605516948, -0.5890489142699] )
    # # anchor angles to hit the ball
    # angle_anchor_ref = np.array( [ 2.94397627, -1.452987321865, -0.87660612618] )
    # # after calibration
    # angle_initial = angle_initial_ref - offset
    # angle_anchor = angle_anchor_ref - offset

    # (_, position_anchor) = Robot.AngleToEnd(angle_anchor, frame='Cartesian')
    # position_error = position_anchor - position_mean
    # position_list = position_list + position_error
    
    # l_1 = 0.4
    # l_2 = 0.38
    # index_list = range(len(time_list))

    # for index in index_list:
    #     position = position_list[index, :] # x, y, z
    #     angle = np.array([0, 0, 0]) / 180 * math.pi
    #     if np.linalg.norm(position, ord=2) <= l_1+l_2:
    #         T = time_list[index]
    #         V = velocity_list[index]
    #         target = Robot.EndToAngle( position, frame='Cartesian') # theta1, theta2, theta3
    #         t_list = np.array([0, T, T+1.0, T+1.1])
    #         (p_mja, p_mjp, p_angular_mja, p_angular_mjp, p_angular_mjl, t_stamp) = Robot.PathPlanning( time_point=0, T_go=T, T_back=1.0, T_steady=0.2,
    #                                                         angle=angle, velocity_initial=np.array([0, 0, 0]), 
    #                                                         acceleration_initial=np.array([0, 0, 0]),
    #                                                         target=target, frequency=100 )
    #         GetPlot(p_mja, p_mjp, p_angular_mja, p_angular_mjp, p_angular_mjl, t_list, 0.01, index)
    RG = RobotGeometry()
    (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=50, angle=[-0.1981112,0.99263366,0.65627032], velocity_initial=[-0.27871242,0.17159836,-0.28683371], acceleration_initial=[6.18824994,-0.04379154,0.07319935], T_go=1.1099999999999999, target=[0.98466532,1.23982882,0.36010159])
    # (p, v, a, j, theta, t_stamp) = RG.PathPlanning(time_point=0, angle=[-0.1981112,0.99263366,0.65627032], T_go=1.1099999999999999-0.5, target=[0.98466532,1.23982882,0.36010159])
    
    legend_position = 'lower right'
    fig = plt.figure(figsize=(18, 18))

    ax_position0 = fig.add_subplot(311)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Position of Dof_0 in degree')
    line = []
    line_temp, = ax_position0.plot(t_stamp, theta[0, :] * 180 / math.pi, linewidth=2, linestyle='dashed', label=r'Pos_Dof0_des')
    line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)
        
    ax_position1 = fig.add_subplot(312)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Position of Dof_1 in degree')
    line = []
    line_temp, = ax_position1.plot(t_stamp, theta[1, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof1_des')
    line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)
    
    ax_position2 = fig.add_subplot(313)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Position of Dof_2 in degree')
    line = []
    line_temp, = ax_position2.plot(t_stamp, theta[2, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof2_des')
    line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)

    plt.suptitle('Joint Space Trajectory Tracking Performance')               
    plt.show()