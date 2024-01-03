'''
This script is used to check the performance of the pid controller.
'''
import numpy as np
import o80
import os
from get_handle import get_handle
import PID_LIB
from SingleJoint import Joint
import matplotlib.pyplot as plt
import math
# %%
# def pressure_initialization(frontend, ago, ant):
#     duration         = o80.Duration_us.seconds(5)
#     frontend.add_command(ago, ant, duration, o80.Mode.QUEUE)
#     frontend.pulse_and_wait()
# %%
dof = 0
task_list = ['StepSignal'] #, 'CraggedSignal'] #, 'SmoothSignal_1', 'SmoothSignal_fast', 'SmoothSignal_hhf', 'SmoothSignal_two']
task = task_list[0]
amp_list = [3]
anchor_ago_list = np.array([17500, 18500, 16000, 15000])
anchor_ant_list = np.array([17500, 18500, 16000, 15000])
anchor_max_list = np.array([22000, 22000, 22000, 22000])
anchor_max_list[dof] = anchor_ago_list[dof]
anchor_ago = anchor_ago_list[dof]
anchor_ant = anchor_ant_list[dof]
# %% get the handle
handle           = get_handle(mode='pressure')
frontend         = handle.frontends["robot"]
# %% build the joints
y_des_list = [[None] * len(task_list) ] * len(amp_list)
y_out_list = [[None] * len(task_list) ] * len(amp_list)
t_stamp_list = [[None] * len(task_list) ] * len(amp_list)
joint = Joint(frontend, dof, anchor_ago, anchor_ant)
row = 0
for amp in amp_list:
    pid = PID_LIB.get_pid_controller(dof, amp)
    joint.pid = pid
    col = 0
    for task in task_list:
        print(task)
        (t_stamp, y_des) = PID_LIB.get_task(task=task, amp=30*math.pi/180)
        # pressure_initialization(frontend=frontend, ago=anchor_max_list, ant=anchor_max_list)
        (y_out, fb, ago, ant, fb_kp, fb_ki, fb_kd, d1, d2, d3) = joint.Control(y=y_des, mode_name="fb", mode_trajectory="ref",
                                                                               ifplot="no", echo="yes")
        
        print(frontend.latest().get_iteration())
        # y_des_list[row][col] = y_des
        # y_out_list[row][col] = y_out
        # t_stamp_list[row][col] = t_stamp

        # plt.plot(t_stamp, fb)
        # plt.show()



        y_out = np.array(y_out)
        legend_position = 'lower right'
        fig, axs = plt.subplots(7, 1, figsize=(40, 20))
        ax = axs[0]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Angle $\theta$ in degree')
        line = []
        line_temp, = ax.plot(t_stamp, (y_des+y_out[0])*180/math.pi, 
                            linewidth=1.5, linestyle='--', label=r'des')
        line.append( line_temp )
        line_temp, = ax.plot(t_stamp, y_out*180/math.pi,
                            linewidth=1, label=r'out')
        line.append( line_temp )
        ax.grid()
        ax.legend(handles=line, loc=legend_position)

        ax = axs[1]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Normalized pressure')
        line = []
        line_temp, = ax.plot(t_stamp, ago, 
                            linewidth=1, label=r'ago')
        line.append( line_temp )
        line_temp, = ax.plot(t_stamp, ant,
                            linewidth=1, label=r'ant')
        line.append( line_temp )
        ax.grid()
        ax.legend(handles=line, loc=legend_position)            
        
        ax = axs[2]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Feedback')
        line = []
        line_temp, = ax.plot(t_stamp, fb, 
                            linewidth=1, label=r'fb')
        line.append( line_temp )
        ax.grid()
        ax.legend(handles=line, loc=legend_position)      

        ax = axs[3]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Feedback')
        line = []
        line_temp, = ax.plot(t_stamp, fb_kp, 
                            linewidth=1, label=r'kp')
        line.append( line_temp )
        line_temp, = ax.plot(t_stamp, fb_ki, 
                            linewidth=1, label=r'ki')
        line.append( line_temp )
        line_temp, = ax.plot(t_stamp, fb_kd, 
                            linewidth=1, label=r'kd')
        line.append( line_temp )
        ax.grid()
        ax.legend(handles=line, loc=legend_position)     

        ax = axs[4]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Ohters')
        line = []
        line_temp, = ax.plot(t_stamp, d1, 
                            linewidth=1, label=r'\delta \theta')
        line.append( line_temp )
        ax.grid()
        ax.legend(handles=line, loc=legend_position)  

        ax = axs[5]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Ohters')
        line = []
        line_temp, = ax.plot(t_stamp, d2, 
                            linewidth=1, label=r'res_i')
        line.append( line_temp )
        ax.grid()
        ax.legend(handles=line, loc=legend_position)  

        ax = axs[6]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Ohters')
        line = []
        line_temp, = ax.plot(t_stamp, d3, 
                            linewidth=1, label=r'\delta \theta')
        line.append( line_temp )
        ax.grid()
        ax.legend(handles=line, loc=legend_position)  
        
        
        plt.show() 


        col += 1
    row += 1
# %%
# legend_position = 'lower right'
# fig, axs = plt.subplots(len(task_list), len(amp_list), figsize=(40, 20))
# for i_row in range(len(task_list)):
#     for i_col in range(len(amp_list)):
#         y_des = y_des_list[i_row][i_col]
#         y_out = y_out_list[i_row][i_col]
#         t_stamp = t_stamp_list[i_row][i_col]

#         ax = axs[i_row, i_col]
#         ax.set_xlabel(r'Time $t$ in s')
#         ax.set_ylabel(r'Angle $\theta$ in degree')
#         line = []
#         line_temp, = ax.plot(t_stamp, y_des*180/math.pi, 
#                             linewidth=1.5, linestyle='--', label=r'des')
#         line.append( line_temp )

#         line_temp, = ax.plot(t_stamp, (y_out-y_out[0])*180/math.pi,
#                             linewidth=1, label=r'out')
#         line.append( line_temp )
#         ax.grid()
#         ax.legend(handles=line, loc=legend_position)     
# plt.show()   