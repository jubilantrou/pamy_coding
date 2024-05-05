'''
This script is used to do the system identification by exciting all the DoFs together
using the random multisine signals whose phases are properly shited by a DFT matrix.
'''
import PAMY_CONFIG
import math
import matplotlib.pyplot as plt
from get_handle import get_handle
import o80_pam
from RealRobotGeometry import RobotGeometry
import numpy as np
import o80
from get_handle import get_handle
import o80_pam

def SystemIdentification(robot, initial_angle, f_input, f_output):
        init_pressures = np.array(frontend.latest().get_observed_pressures())
        init_positions = np.array(frontend.latest().get_positions())

        amp_list = [4.0, 3.0, 3.0, 2.0]
        print('amp list: {}'.format(amp_list))
       
        frequency              = 500.0
        period                 = 1.0/frequency
        fs                     = 100
        duration_per_command   = 1 / fs
        iterations_per_command = int(duration_per_command/period)

        u         = np.loadtxt(f_input, delimiter=',')
        p         = 10 # p still indicates how many periods used for repeat
        [m, N]    = u.shape # now m is a constant 3 corresponding to 3 signals for DoF 1-3, while N is still the number of time points in each signal
        u         = np.tile(u, p)
        t_stamp_u = np.arange(0, (N*p)/fs, 1/fs)
        t         = 1 / fs
        T         = (N*p) / fs

        ref_iteration   = frontend.latest().get_iteration()
        iteration_begin = ref_iteration + 500
        iteration       = iteration_begin

        robot.frontend.add_command(robot.anchor_ago_list, robot.anchor_ant_list,
                                  o80.Iteration(iteration_begin-1),
                                  o80.Mode.QUEUE)
        robot.frontend.pulse()

        for i in range(3):
            u[i,:] = u[i,:]*amp_list[i]

        total_iter = N*p
        for i in range(total_iter):          
            pressure_ago = np.array([], dtype=int)
            pressure_ant = np.array([], dtype=int)

            for dof in range(3):
                diff = u[dof, i]
                pressure_ago = np.append(pressure_ago, int( robot.anchor_ago_list[dof] + diff ))
                pressure_ant = np.append(pressure_ant, int( robot.anchor_ant_list[dof] - diff ))
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

        iteration_end  = iteration
        iteration = iteration_begin

        position = np.array([])
        velocity = np.array([])
        obs_pressure_ant = np.array([])
        obs_pressure_ago = np.array([])
        des_pressure_ant = np.array([])
        des_pressure_ago = np.array([])
        t_stamp = np.array([])

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
            position         = np.append(position, obs_position)
            velocity         = np.append(velocity, obs_velocity)
            t_stamp          = np.append(t_stamp, observation.get_time_stamp() * 1e-9)

            iteration += iterations_per_command
        
        initial_time = t_stamp[0]
        for i in range(0, len(t_stamp)):
            t_stamp[i] = t_stamp[i] - initial_time

        # use a simple plotting to make sure DoF2 and DoF3 will not touch the boundary very often,
        # i.e. reaching 90 degrees, otherwise we need to reduced used amp values
        plt.plot(t_stamp, [position[4*i+0]/math.pi*180 for i in range(total_iter)])
        plt.plot(t_stamp, [position[4*i+1]/math.pi*180 for i in range(total_iter)], color='g')
        plt.plot(t_stamp, [position[4*i+2]/math.pi*180 for i in range(total_iter)], color='y')
        plt.axhline(y=90, color='r', linestyle='-')
        plt.show()

        print("total duration:",t_stamp[-1]-t_stamp[0])
        print("expected duration:",t_stamp_u[-1])
        print("number of collected points:",len(t_stamp))
        print("desired number of collected points:",len(t_stamp_u))

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
        np.savetxt(f, init_positions,   fmt='%.8f')
        np.savetxt(f, init_pressures,   fmt='%.8f')
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

    f_input = '/home/mtian/Pamy_OCO/excitation signals/excitation_3Hz_DFT_short_III_3.csv'
    f_output = "/home/mtian/Desktop/MPI-intern/" + "test_DFT_short_III_3_SI_result_real_robot_3Hz" + ".txt"
    # TODO: also need to ensure that the robot is already at its desired initial setting before starting the SI, either by 
    # adjusting it manually after pressure initialization or by using the PID/LQR controller that you have tuned
    SystemIdentification(robot=Pamy, initial_angle=PAMY_CONFIG.GLOBAL_INITIAL, f_input=f_input, f_output=f_output)
