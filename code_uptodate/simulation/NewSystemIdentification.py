import PAMY_CONFIG
import math
import os
import matplotlib.pyplot as plt
from get_handle import get_handle
import o80_pam
from RealRobotGeometry import RobotGeometry
import scipy
import numpy as np
import o80
import os
from get_handle import get_handle
import o80_pam

def SystemIdentification(robot, initial_angle, file_name):
        robot.AngleInitialization(angle=initial_angle)
        reset_pressures = np.array(robot.frontend.latest().get_observed_pressures())
        robot.anchor_ago_list = reset_pressures[:, 0]
        robot.anchor_ant_list = reset_pressures[:, 1]

        ago_pressure_max = PAMY_CONFIG.ago_max_list - robot.anchor_ago_list
        ago_pressure_min = PAMY_CONFIG.ago_min_list - robot.anchor_ago_list
        ant_pressure_max = PAMY_CONFIG.ant_max_list - robot.anchor_ant_list
        ant_pressure_min = PAMY_CONFIG.ant_min_list - robot.anchor_ant_list
        amp_list = [min([ago_pressure_max[i],-ago_pressure_min[i],ant_pressure_max[i],-ant_pressure_min[i]])/1000 for i in range(len(PAMY_CONFIG.dof_list))]
        print(amp_list)

        fs                     = 100
        frequency              = 500.0
        period                 = 1.0/frequency
        duration_per_command   = 1 / fs
        iterations_per_command = int(duration_per_command/period)

        u         = np.loadtxt(file_name, delimiter=',')
        p         = 1
        [m, N]    = u.shape
        t_stamp_u = np.arange(0, N/fs, 1/fs)
        t         = 1 / fs
        T         = N / fs

        ref_iteration   = frontend.latest().get_iteration()
        iteration_begin = ref_iteration + 500
        iteration       = iteration_begin

        for i in range(3):
            u[i,:] = u[i,:]*amp_list[i]

        for i in range(N):            
            pressure_ago = np.array([], dtype=int)
            pressure_ant = np.array([], dtype=int)

            pressure_ago = np.append(pressure_ago, int( robot.anchor_ago_list[0] ))
            pressure_ant = np.append(pressure_ant, int( robot.anchor_ant_list[0] ))

            for dof in range(1):
                diff = u[dof+1, i]
                pressure_ago = np.append(pressure_ago, int( robot.anchor_ago_list[dof] + diff ))
                pressure_ant = np.append(pressure_ant, int( robot.anchor_ant_list[dof] - diff ))
            
            pressure_ago = np.append(pressure_ago, int( robot.anchor_ago_list[2] ))
            pressure_ant = np.append(pressure_ant, int( robot.anchor_ant_list[2] ))
            pressure_ago = np.append(pressure_ago, int( robot.anchor_ago_list[3] ))
            pressure_ant = np.append(pressure_ant, int( robot.anchor_ant_list[3] ))

            robot.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)    
            robot.frontend.pulse()
            robot.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)
            robot.frontend.pulse_and_wait()
            
            iteration += iterations_per_command

        f_output = "/home/mtian/Desktop/MPI-intern/" + "new_SI_result_real_robot_3Hz_demo" + ".txt"
        position = np.array([])
        velocity = np.array([])
        obs_pressure_ant = np.array([])
        obs_pressure_ago = np.array([])
        des_pressure_ant = np.array([])
        des_pressure_ago = np.array([])
        t_stamp = np.array([])

        iteration_end  = iteration
        iteration = iteration_begin
        while iteration < iteration_end:
            observation = frontend.read(iteration)

            obs_pressure = np.array(observation.get_observed_pressures())
            des_pressure = np.array(observation.get_desired_pressures())
            obs_position = np.array(observation.get_positions())
            obs_velocity = np.array(observation.get_velocities())

            obs_pressure_ago = np.append(obs_pressure_ago, obs_pressure[:,0])
            obs_pressure_ant = np.append(obs_pressure_ant, obs_pressure[:,1])

            des_pressure_ago = np.append(des_pressure_ago, des_pressure[:,0])
            des_pressure_ant = np.append(des_pressure_ant, des_pressure[:,1])

            position = np.append(position, obs_position)
            velocity = np.append(velocity, obs_velocity)

            t_stamp = np.append(t_stamp, observation.get_time_stamp() * 1e-9)

            iteration += iterations_per_command
        
        initial_time = t_stamp[0]
        for i in range(0, len(t_stamp)):
            t_stamp[i] = t_stamp[i] - initial_time

        print("total duration:",t_stamp[-1]-t_stamp[0])
        print("expected:",t_stamp_u[-1])
        print("number of simulation:",len(t_stamp))
        print("desired number:",len(t_stamp_u))

        print("begin to write data to the file...")
        f = open(f_output, 'w')
        f.write(str(p))
        f.write("\n")
        f.write(str(m))
        f.write("\n")
        f.write(str(N))
        f.write("\n")
        np.savetxt(f, t_stamp,          fmt='%.5f')
        np.savetxt(f, u,                fmt='%.8f')
        np.savetxt(f, des_pressure_ago, fmt='%.5f')
        np.savetxt(f, des_pressure_ant, fmt='%.5f')
        np.savetxt(f, obs_pressure_ago, fmt='%.5f')
        np.savetxt(f, obs_pressure_ant, fmt='%.5f')
        np.savetxt(f, position,         fmt='%.8f')
        f.close()
        print('...completed')

if __name__=='__main__':
    obj = 'real'

    if obj=='sim':
        handle   = get_handle()
        frontend = handle.frontends["robot"]
    elif obj=='real':
        frontend = o80_pam.FrontEnd("real_robot")
    Pamy = PAMY_CONFIG.build_pamy(frontend=frontend)
    RG = RobotGeometry(initial_posture=PAMY_CONFIG.GLOBAL_INITIAL)

    file_name = '/home/mtian/Pamy_OCO/excitation signals/excitation_10Hz_new.csv'
    SystemIdentification(robot=Pamy, initial_angle=PAMY_CONFIG.GLOBAL_INITIAL, file_name=file_name)
