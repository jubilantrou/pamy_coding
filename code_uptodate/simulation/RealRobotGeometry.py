'''
This script is used to define the related functions 
for doing coordinate transformation and planning the trajectory using the minimum jerk algorithm.
'''
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MinJerk_penalty as MJ_penalty
import random
import PAMY_CONFIG
from OCO_utils import *

class RobotGeometry:

    def __init__(self, center_of_robot=[0, 0, 0], x_of_robot=[1, 0, 0],
                 y_of_robot=[0, 1, 0], z_of_robot=[0, 0, 1],
                 l_1=0.40, l_2=0.38, step_size=0.01, initial_posture=None, constraint_angle=None):
        # for now the center of the robot must be [0, 0, 0]
        # TODO: need to make the variable center_of_robot changable later, incorporate it into functions of RobotGeometry class, and keep it be in accordance with the real setting
        self.center_of_robot = center_of_robot
        self.x_of_robot = x_of_robot
        self.y_of_robot = y_of_robot
        self.z_of_robot = z_of_robot
        
        if initial_posture is not None:
            self.initial_posture = initial_posture
        else:
            raise ValueError("The initial posture must be given, using absolute values in joint space in rad, in the form of np.array([value_1, value_2, value_3, value_4]).")
        self.l_1 = l_1
        self.l_2 = l_2
        self.step_size = step_size
        self.constraint_angle = constraint_angle
        (_, self.position_final) = self.AngleToEnd(self.initial_posture[0:3], frame='Cylinder')
        self.acceleration_final = [0, 0, 0]

    # def UpdateInitial(angle):
    #     self.initial_posture = angle
    #     (_, self.position_final) = self.AngleToEnd(self.initial_posture[0:3], frame='Cylinder')

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
            x = np.cos(theta0) * np.sin(theta1) * l_1
            y = np.sin(theta0) * np.sin(theta1) * l_1
            z = np.cos(theta1) * l_1
            position_A = np.array([-y, x, z+1.21])
            x = np.cos(theta0) * np.sin(theta1) * l_1 + np.cos(theta0) * np.sin(theta1 + theta2) * l_2
            y = np.sin(theta0) * np.sin(theta1) * l_1 + np.sin(theta0) * np.sin(theta1 + theta2) * l_2
            z = np.cos(theta1) * l_1 + np.cos(theta1 + theta2) * l_2
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
            position_A = np.array([theta0, abs(l_1*math.sin(theta1)), (l_1*math.cos(theta1))+1.21])
            r = np.sqrt( l_1**2 + l_2**2 - 2*l_1*l_2*math.cos(math.pi-abs(theta2)) )
            gama = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
            if theta2 > 0:
                alpha = theta1 + gama
            else:
                alpha = theta1 - gama
            position_B = np.array([theta0, abs(r*math.sin(alpha)), (r*math.cos(alpha))+1.21])

        return (position_A, position_B)

    def EndToAngle(self, position, frame='Cartesian'):
        l_1 = self.l_1
        l_2 = self.l_2
        # transform coordinates of the second robot arm's end effector in Cartesian space into joint space
        if frame == 'Cartesian':
            x = position[1]
            y = -position[0]
            z = position[2]-1.21

            l      = np.linalg.norm((x,y,z), ord=2)
            theta0 = math.atan2(y,x)

            temp = (l_1**2 + l**2 - l_2**2) / (2*l_1*l)
            if temp>1:
                temp=1
            if temp<-1:
                temp=-1
            gamma  = math.acos(temp)
            alpha  = math.asin(z/l)
            theta1 = math.pi/2 - gamma - alpha

            # the computation of theta2 below is under one assumption that
            # theta2 can not take negative values, which is true for almost all the interception movements
            # TODO: need to make sure the chosen interception policy satisfies this assumption
            temp1 = (l_1**2 + l_2**2 - l**2) / (2*l_1*l_2)
            if temp1>1:
                temp1=1
            if temp1<-1:
                temp1=-1
            beta   = math.acos(temp1)
            theta2 = math.pi - beta

            angle  = np.array([theta0, theta1, theta2])
        return angle

    def CalAngularTrajectory(self, trajectory, angle, frame='Cartesian'):
        # transform trajectory of the second robot arm's end effector into joint space
        l_1 = self.l_1
        l_2 = self.l_2

        angle_trajectory = np.array([])

        # only the code for Cylinder space is checked
        # TODO: need to check the codes for the other two spaces
        if frame == 'Cylinder':
            # dimension of trajectory: 3 x length
            angle_trajectory = np.zeros(trajectory.shape)

            for i in range(trajectory.shape[1]):
                theta0 = trajectory[0, i]
                l = trajectory[1, i]
                z = trajectory[2, i]-1.21
                theta0_ = theta0
                theta2 = angle[2]

                r = np.sqrt( l**2 + z**2 ) if np.sqrt( l**2 + z**2 ) <= l_1+l_2 else l_1+l_2

                if abs(z/r) <= 1:
                    z_over_r = z/r
                elif z/r < 0:
                    z_over_r = -1
                elif z/r > 0:
                    z_over_r = 1
                # alpha = -abs( math.pi/2 - math.asin( z_over_r) )
                alpha = abs( math.pi/2 - math.asin( z_over_r) )

                if abs ((l_1**2 + l_2**2 - r**2) / (2*l_1*l_2)) <= 1:
                    l_l = (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2)
                elif (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) > 0:
                    l_l = 1
                elif (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) < 0:
                    l_l = -1
                theta2_ = math.pi - math.acos( l_l )
                # make the sign of theta2_ the same as the one of theta2
                if abs(-theta2_ - theta2) < abs(theta2_ - theta2):
                    theta2_ = -theta2_

                theta_temp = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
                if theta2_ > 0:
                    theta1_ = alpha - theta_temp  
                else:
                     theta1_ = alpha + theta_temp
    
                angle_trajectory[:, i] = np.array([theta0_, theta1_, theta2_])   

        # elif frame == 'Cartesian':
        #     for i in range(trajectory.shape[1]):
        #         x = trajectory[0, i]
        #         y = trajectory[1, i]
        #         z = trajectory[2, i]

        #         theta0 = angle[0]
        #         theta1 = angle[1]
        #         theta2 = angle[2]

        #         l = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        #         if l > l_1 + l_2:
        #             x = (l_1 + l_2) / l * x
        #             y = (l_1 + l_2) / l * y
        #             z = (l_1 + l_2) / l * z

        #         cos_theta0 = x / np.sqrt((x ** 2 + y ** 2))
        #         sin_theta0 = y / np.sqrt((x ** 2 + y ** 2))

        #         theta0_ = math.acos( cos_theta0 )
        #         if sin_theta0 < 0:
        #             theta0_ = - abs(theta0_)
                
        #         if cos_theta0 != 0:
        #             x_tilde = x / cos_theta0
        #         else:
        #             x_tilde = y / sin_theta0
                
        #         l_tilde = np.sqrt( z**2 + x_tilde**2 )

        #         theta2_temp = math.acos( - round((l_tilde ** 2 - l_1 ** 2 - l_2 ** 2) / (2 * l_1 * l_2), 4) )
        #         theta2_ = math.pi - theta2_temp
        #         if abs(-theta2_ - theta2) < abs(theta2_ - theta2):
        #             theta2_ = -theta2_

        #         theta1_temp1 = math.acos( -round((l_2 ** 2 - l_1 ** 2 - l_tilde ** 2) / (2 * l_1 * l_tilde), 4) )
        #         theta1_temp2 = math.asin( z / l_tilde )
        #         if theta2_ < 0:
        #             theta1_ = math.pi / 2 - theta1_temp2
        #         else:
        #             theta1_ = math.pi / 2 - theta1_temp1 - theta1_temp2

        #         angle = np.array([theta0_, theta1_, theta2_])

        #         angle_trajectory = np.append(angle_trajectory, angle)
        
        # elif frame == 'Polar':
        #     for i in range(trajectory.shape[1]):
        #         theta0 = trajectory[0, i]
        #         alpha = trajectory[1, i]
        #         r = trajectory[2, i]

        #         theta1 = angle[1]
        #         theta2 = angle[2]

        #         theta0_ = theta0

        #         theta2_ = math.pi - math.acos( (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) )

        #         if abs(-theta2_ - theta2) < abs(theta2_ - theta2):
        #             theta2_ = -theta2_

        #         theta_temp = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
        #         if theta2_ > 0:
        #             theta1_ = alpha - theta_temp
        #         else:
        #             theta1_ = alpha + theta_temp

        #         angle = np.array([theta0_, theta1_, theta2_])

        #         angle_trajectory = np.append(angle_trajectory, angle)  

        return angle_trajectory

    def updatedPathPlanning(self, time_point, T_go=1.0, angle=None, target=None, method=None, target_vel=None):
        '''
        to mimic the reference trajectory with multiple online updates based on the function PathPlanning()

        Args:
            time_point: the number of time steps corresponding to the start time stamp
            T_go: the time duration from starting to intercepting the ball
            angle: the start position in joint space
            target: the target interception position in joint space
            method: 'no_delay' indicates updating the trajectory directly, which may lead to mutations in extracted data later, while 'with_delay' indicates updating the trajectory with a delay(10 time steps i.e. 0.1s for now), 
                    which doesn't lead to mutations in extracted data but constrain the choice of part of the sliding window size later
                    (we stick to the method 'no_delay' now)
            target_vel: the target linear velocity of the ball after the interception, which will be randomly generated when without a specified value
        Returns:
            p_final: positions of all DoFs at different time points with all updates included, for the second robot arm's end effector in Cylinder space
            v_fianl: velocities of all DoFs at different time points with all updates included, for the second robot arm's end effector in Cylinder space
            a_final: accelerations of all DoFs at different time points with all updates included, for the second robot arm's end effector in Cylinder space
            j_final: jerks for of DoFs at different time points with all updates included, for the second robot arm's end effector in Cylinder space
            theta_final: transformed positions of all DoFs at different time points with all updates included, for the joint space
            t_stamp_final: an array storing all the time stamps got from spacing the entire duration evenly using the control period
            vel: the finally used target linear velocity of the ball after the interception 
            theta_list: a list with each element storing the joint space positions of each replanning
            t_stamp_list: a list with each element storing the time stamps of each replanning
            p_int_record: a list with each element storing the target interception position of each replanning
            T_go_list: a list with each element storing the target interception time point of each replanning
            time_update_record: a list storing the number of how many time steps of each replanning are included in the final trajectory, i.e. recording the local time stamp of each replanning when the update happens
            update_point_index_list: a list storing the accumulated number in time_update_record, i.e. recording the global time stamps of the final updated planning when the updates happen
        '''
        # below lists used to store the full results of each replanning
        p_list = []
        v_list = []
        a_list = []
        j_list = []
        theta_list = []
        t_stamp_list = []
        # below lists used to store the updates information
        p_int_record = [target]
        T_go_list = [T_go]
        time_update_record = []

        (p, v, a, j, theta, t_stamp, vel) = self.PathPlanning(time_point=time_point, T_go=T_go, angle=angle, target=target, part=0, target_vel=target_vel)
        print('desired interpetion vel. is: {}'.format(vel))
        p_list.append(p)
        v_list.append(v)
        a_list.append(a)
        j_list.append(j)
        theta_list.append(theta)
        t_stamp_list.append(t_stamp)

        # start updating after a certain amount of time, 0.3s here, as the estimated states of ball are not accurate enough at initial stage
        idx_begin = 30
        t_begin = 0.3
        # stop updating when there is not long enough time interval to the interception time stamp, 0.4s here, considering that replanning takes time and 
        # that estimated states of ball after the contatct with table are rather accurate
        while t_stamp[-1]-t_begin>=0.4:
            # update the interception time stamp randomly, but constrain it within the interval [0.85, 1.1]
            T_go += random.randrange(-5, 5)/100
            if T_go>1.1:
                T_go = 1.1
            elif T_go<0.85:
                T_go = 0.85
            
            # update the interception position, by randomly changing the second robot arm's end effector in Cartesian space that it is corresponding to, 
            # but constrain it within certain intervals
            (_, temp) = self.AngleToEnd(target[0:3], frame='Cartesian')
            temp += [random.randrange(-3,3)/100,random.randrange(-3,3)/100,random.randrange(-3,3)/100]
            target = self.EndToAngle(temp)
            if target[0] > 90/180*math.pi:
                target[0] = 90/180*math.pi
            elif target[0] < -90/180*math.pi:
                target[0] = -90/180*math.pi
            if target[1] > 80/180*math.pi:
                target[1] = 80/180*math.pi
            elif target[1] < 45/180*math.pi:
                target[1] = 45/180*math.pi
            if target[2] > 75/180*math.pi:
                target[2] = 75/180*math.pi
            elif target[2] < 15/180*math.pi:
                target[2] = 15/180*math.pi

            p_int_record.append(target)
            T_go_list.append(T_go)

            # here we simply assume that updates happen every 0.1s
            if T_go-(t_begin+0.1) >= 0.4:
                if method=='with_delay':
                    (p, v, a, j, theta, t_stamp, _) = self.PathPlanning(time_point=(t_begin*100+10), T_go=T_go, angle=theta[:,(idx_begin+10)], velocity_initial=v[:,(idx_begin+10)], 
                                                                        acceleration_initial=a[:,(idx_begin+10)], target=target, part=0, target_vel=vel)
                    time_update_record.append(idx_begin+10)
                    idx_begin = 0
                    t_begin += 0.1
                elif method=='no_delay':
                    (p, v, a, j, theta, t_stamp, _) = self.PathPlanning(time_point=(t_begin*100), T_go=T_go, angle=theta[:,(idx_begin)], velocity_initial=v[:,(idx_begin)], 
                                                                        acceleration_initial=a[:,(idx_begin)], target=target, part=0, target_vel=vel)
                    time_update_record.append(idx_begin)
                    idx_begin = 10
                    t_begin += 0.1
                    
                p_list.append(p)
                v_list.append(v)
                a_list.append(a)
                j_list.append(j)
                theta_list.append(theta)
                t_stamp_list.append(t_stamp)

            else:
                if method=='with_delay':
                    (p, v, a, j, theta, t_stamp, _) = self.PathPlanning(time_point=(t_begin*100+10), T_go=T_go, angle=theta[:,(idx_begin+10)], velocity_initial=v[:,(idx_begin+10)], 
                                                                        acceleration_initial=a[:,(idx_begin+10)], target=target, part=0, target_vel=vel)
                    time_update_record.append(idx_begin+10)
                elif method=='no_delay':
                    (p, v, a, j, theta, t_stamp, _) = self.PathPlanning(time_point=(t_begin*100), T_go=T_go, angle=theta[:,(idx_begin)], velocity_initial=v[:,(idx_begin)], 
                                                                        acceleration_initial=a[:,(idx_begin)], target=target, part=0, target_vel=vel)
                    time_update_record.append(idx_begin)
                
                p_list.append(p)
                v_list.append(v)
                a_list.append(a)
                j_list.append(j)
                theta_list.append(theta)
                t_stamp_list.append(t_stamp)

                break

        p_final       = np.hstack([p_list[i][:,:time_update_record[i]]     for i in range(len(time_update_record))] + [p_list[-1]])
        v_final       = np.hstack([v_list[i][:,:time_update_record[i]]     for i in range(len(time_update_record))] + [v_list[-1]])
        a_final       = np.hstack([a_list[i][:,:time_update_record[i]]     for i in range(len(time_update_record))] + [a_list[-1]])
        j_final       = np.hstack([j_list[i][:,:time_update_record[i]]     for i in range(len(time_update_record))] + [j_list[-1]])
        theta_final   = np.hstack([theta_list[i][:,:time_update_record[i]] for i in range(len(time_update_record))] + [theta_list[-1]])
        t_stamp_final = np.hstack([t_stamp_list[i][:time_update_record[i]] for i in range(len(time_update_record))] + [t_stamp_list[-1]])

        update_point_index_list = []
        num = 0
        for i in range(len(time_update_record)):
            num += time_update_record[i]
            update_point_index_list.append(num)
        
        return (p_final, v_final, a_final, j_final, theta_final, t_stamp_final, vel, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list)

    def PathPlanning(self, time_point, T_go=1.0, T_back=1.35, T_steady=0.15,
                     angle=None, velocity_initial=np.array([0, 0, 0]), acceleration_initial=np.array([0, 0, 0]),
                     target=None, frequency=100, plan_weight=(12.5, 5.5), part=1, target_vel=None):
        '''
        to generate the reference trajectory using the minimum jerk algorithm under given conditions 

        Args:
            time_point: the number of time steps corresponding to the start time stamp
            T_go: the time duration from starting to intercepting the ball
            T_back: the time duration from intercepting the ball to getting back to the initial position
            T_steady: the time duration from getting back to the initial position to getting steady at the initial position
            angle: the start position in joint space
            velocity_initial: the strat velocity
            acceleration_initial: the strat acceleration
            target: the target interception position in joint space
            frequency: the control frequency
            plan_weight: tunable parameters for the minimum jerk algorithm
            part: 1 indicates only planning the part until the interception(need to make sure time_point < round(T_go * frequency) by yourself for this case), while 0 indicates planning the whole process
            target_vel: the target linear velocity of the ball after the interception, which will be randomly generated when without a specified value
        Returns:
            p_mjp: computed positions of all DoFs at different time points, for the second robot arm's end effector in Cylinder space
            p_mjv: computed velocities of all DoFs at different time points, for the second robot arm's end effector in Cylinder space
            p_mja: computed accelerations of all DoFs at different time points, for the second robot arm's end effector in Cylinder space
            p_mjj: computed jerks for of DoFs at different time points, for the second robot arm's end effector in Cylinder space
            p_angular_mjp: transformed positions of all DoFs at different time points, for the joint space
            t_stamp: an array storing all the time stamps got from spacing the entire duration evenly using the control period 
            vel: the finally used target linear velocity of the ball after the interception
        '''
        (_, p_initial) = self.AngleToEnd(angle[0:3], frame='Cylinder')
        (_, p_target) = self.AngleToEnd(target[0:3], frame='Cylinder')

        if target_vel is None:
            vel = random.randrange(40, 60)/10
        else:
            vel = target_vel

        if p_target[0]<=0:
            v_target = np.array([vel/p_target[1], 0, 0])
        else:
            v_target = np.array([-vel/p_target[1], 0, 0])
        v_final  = np.array([0, 0, 0])
        a_target = np.array([0, 0, 0])
        a_final  = np.array([0, 0, 0])
        
        if time_point >= round(T_go * frequency):
            m_list = [[1.0, 1.0],
                      [1.0, 1.0],
                      [1.0, 1.0]]
            m_list = np.array(m_list)
            
            n_list = [[4.0, 0.1],
                      [1.0, 0.1],
                      [1.0, 0.1]]
            n_list = np.array(n_list)

            t = np.array([time_point/frequency, T_go+T_back, T_go+T_back+T_steady])
            p = np.array([p_initial, self.position_final, self.position_final]).T
            p = p - p_initial.reshape(-1, 1)
            v = np.array([velocity_initial, v_final, v_final]).T
            a = np.array([acceleration_initial, a_final, a_final]).T

        elif time_point < round(T_go * frequency):
            # m_list stores the relative terms
            m_list = [[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]]
            m_list = np.array(m_list)

            # tune the terms in n_list to adjust the penalty
            n_list = [[plan_weight[0], plan_weight[1], 0.1],
                      [1.0, 1.0, 0.1],
                      [1.0, 1.0, 0.1]]
            n_list = np.array(n_list)

            t = np.array([time_point/frequency, T_go, T_go+T_back, T_go+T_back+T_steady])
            p = np.array([p_initial, p_target, self.position_final, self.position_final]).T
            p = p - p_initial.reshape(-1, 1)
            v = np.array([velocity_initial, v_target, v_final, v_final]).T
            a = np.array([acceleration_initial, a_target, a_final, a_final]).T
        
        if part:
            [p_mjp, p_mjv, p_mja, p_mjj, t_stamp] = MJ_penalty.PathPlanning(p[:,:2], v[:,:2], a[:,:2], t[:2], 1/frequency, m_list, n_list)
        else:
            [p_mjp, p_mjv, p_mja, p_mjj, t_stamp] = MJ_penalty.PathPlanning(p, v, a, t, 1/frequency, m_list, n_list)
        p_mjp = p_mjp + p_initial.reshape(-1, 1)

        # p_angular_mjp = self.CalAngularTrajectory(p_mjp, angle[0:3] + self.initial_posture[0:3], frame='Cylinder')
        p_angular_mjp = self.CalAngularTrajectory(p_mjp, self.initial_posture[0:3], frame='Cylinder')

        return (p_mjp, p_mjv, p_mja, p_mjj, p_angular_mjp, t_stamp, vel)

if __name__ == '__main__':
    RG = RobotGeometry(initial_posture=PAMY_CONFIG.GLOBAL_INITIAL)

    ###############################
    ### test the penalty parameters
    ###############################

    # fig = plt.figure(figsize=(18, 18))
    # ax1_position0 = fig.add_subplot(111)
    # plt.xlabel(r'Time $t$ in s')
    # plt.ylabel(r'Dof_0')
    # line = []

    # for i in range(200):
    #     (t, angle) = get_random()
    #     (p, v, a, j, theta, t_stamp, _) = RG.PathPlanning(time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, part=0)
    #     line_temp, = ax1_position0.plot(t_stamp, theta[0, :]/math.pi*180, linewidth=1)
    #     line.append( line_temp )

    # # add references to make sure the movement range of DoF1 is not too large for the safety reason
    # line_temp, = ax1_position0.plot(t_stamp, [120]*len(t_stamp), linewidth=1)
    # line.append( line_temp )
    # line_temp, = ax1_position0.plot(t_stamp, [-120]*len(t_stamp), linewidth=1)
    # line.append( line_temp )

    # plt.legend(handles=line, shadow=True)
    # plt.show()

    ###########################################
    ### test the function updatedPathPlanning()
    ###########################################

    t = 0.97
    angle = [0.581, 0.794, 1.037]
    (p, v, a, j, theta, t_stamp, vel, theta_list, t_stamp_list, p_int_record, T_go_list, time_update_record, update_point_index_list) = RG.updatedPathPlanning(
        time_point=0, T_go=t, angle=PAMY_CONFIG.GLOBAL_INITIAL, target=angle, method='no_delay', target_vel=4)

    legend_position = 'best'
    fig = plt.figure(figsize=(18, 18))

    ax_position0 = fig.add_subplot(311)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Position of Dof_0 in degree')
    line = []
    line_temp, = ax_position0.plot(t_stamp, theta[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof0_des')
    line.append( line_temp )
    for j in range(len(theta_list)):
        line_temp, = ax_position0.plot(t_stamp_list[j], theta_list[j][0, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof0_traj_candidate_'+str(j+1))
        line.append( line_temp )
        line_temp, = ax_position0.plot(T_go_list[j], p_int_record[j][0] * 180 / math.pi, 'o', label='target_'+str(j+1))
        line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)
        
    ax_position1 = fig.add_subplot(312)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Position of Dof_1 in degree')
    line = []
    line_temp, = ax_position1.plot(t_stamp, theta[1, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof1_des')
    line.append( line_temp )
    for j in range(len(theta_list)):
        line_temp, = ax_position1.plot(t_stamp_list[j], theta_list[j][1, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof1_traj_candidate_'+str(j+1))
        line.append( line_temp )
        line_temp, = ax_position1.plot(T_go_list[j], p_int_record[j][1] * 180 / math.pi, 'o', label='target_'+str(j+1))
        line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)
    
    ax_position2 = fig.add_subplot(313)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Position of Dof_2 in degree')
    line = []
    line_temp, = ax_position2.plot(t_stamp, theta[2, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof2_des')
    line.append( line_temp )
    for j in range(len(theta_list)):
        line_temp, = ax_position2.plot(t_stamp_list[j], theta_list[j][2, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof2_traj_candidate_'+str(j+1))
        line.append( line_temp )
        line_temp, = ax_position2.plot(T_go_list[j], p_int_record[j][2] * 180 / math.pi, 'o', label='target_'+str(j+1))
        line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)

    plt.suptitle('Joint Space Trajectory Tracking Performance')
    plt.show()
