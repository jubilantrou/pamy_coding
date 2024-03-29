'''
This script is used to define the plotting functions 
for the OCO training procedure.
'''
import math
import matplotlib.pyplot as plt
import wandb

def wandb_plot(i_iter, frequency, t_stamp, ff, fb, y, theta_, t_stamp_list, theta_list, T_go_list, p_int_record, obs_ago, obs_ant, des_ago, des_ant, SI_ref=None, disturbance=None):
    '''
    to plot the ff input, the fb input and the tracking performance in joint space
    '''
    plots = []
    if (i_iter+1)%frequency==0:
        legend_position = 'best'
        fig1 = plt.figure(figsize=(18, 18))

        ax1_position0 = fig1.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_0')
        line = []
        line_temp, = ax1_position0.plot(t_stamp, ff[0, :], linewidth=2, label=r'uff_Dof0')
        line.append( line_temp )
        line_temp, = ax1_position0.plot(t_stamp, fb[0, :], linewidth=2, label=r'ufb_Dof0')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
            
        ax1_position1 = fig1.add_subplot(312)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_1')
        line = []
        line_temp, = ax1_position1.plot(t_stamp, ff[1, :], linewidth=2, label=r'uff_Dof1')
        line.append( line_temp )
        line_temp, = ax1_position1.plot(t_stamp, fb[1, :], linewidth=2, label=r'ufb_Dof1')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
        
        ax1_position2 = fig1.add_subplot(313)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure Input for Dof_2')
        line = []
        line_temp, = ax1_position2.plot(t_stamp, ff[2, :], linewidth=2, label=r'uff_Dof2')
        line.append( line_temp )
        line_temp, = ax1_position2.plot(t_stamp, fb[2, :], linewidth=2, label=r'ufb_Dof2')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Pressure Input'+' Iter '+str(i_iter+1))
        plots.append(wandb.Image(plt, caption="matplotlib image"))                

        fig = plt.figure(figsize=(18, 18))

        ax_position0 = fig.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position of Dof_0 in degree')
        line = []
        line_temp, = ax_position0.plot(t_stamp, y[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof0_out')
        line.append( line_temp )
        line_temp, = ax_position0.plot(t_stamp, theta_[0, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof0_des')
        line.append( line_temp )
        if SI_ref is not None:
            line_temp, = ax_position0.plot(t_stamp, SI_ref[0].reshape(-1) * 180 / math.pi, linewidth=1, linestyle=(0,(5,5)), label=r'Pos_Dof0_SI')
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
        line_temp, = ax_position1.plot(t_stamp, y[1, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof1_out')
        line.append( line_temp )
        line_temp, = ax_position1.plot(t_stamp, theta_[1, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof1_des')
        line.append( line_temp )
        if SI_ref is not None:
            line_temp, = ax_position1.plot(t_stamp, (SI_ref[1].reshape(-1) * 180 / math.pi +45), linewidth=1, linestyle=(0,(5,5)), label=r'Pos_Dof1_SI')
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
        line_temp, = ax_position2.plot(t_stamp, y[2, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof2_out')
        line.append( line_temp )
        line_temp, = ax_position2.plot(t_stamp, theta_[2, :] * 180 / math.pi, linewidth=2, label=r'Pos_Dof2_des')
        line.append( line_temp )
        if SI_ref is not None:
            line_temp, = ax_position2.plot(t_stamp, (SI_ref[2].reshape(-1) * 180 / math.pi + 40), linewidth=1, linestyle=(0,(5,5)), label=r'Pos_Dof2_SI')
            line.append( line_temp )
        for j in range(len(theta_list)):
            line_temp, = ax_position2.plot(t_stamp_list[j], theta_list[j][2, :] * 180 / math.pi, linewidth=2, linestyle=(0,(5,5)), label='Dof2_traj_candidate_'+str(j+1))
            line.append( line_temp )
            line_temp, = ax_position2.plot(T_go_list[j], p_int_record[j][2] * 180 / math.pi, 'o', label='target_'+str(j+1))
            line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Joint Space Trajectory Tracking Performance'+' Iter '+str(i_iter+1))
        plots.append(wandb.Image(plt, caption="matplotlib image"))

        ### added pressure fig
        fig2 = plt.figure(figsize=(18, 18))

        p_position0 = fig2.add_subplot(311)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressures of Dof_0')
        line = []
        line_temp, = p_position0.plot(t_stamp, obs_ago[0,:], linewidth=2, label=r'obs_ago')
        line.append( line_temp )
        line_temp, = p_position0.plot(t_stamp, obs_ant[0,:], linewidth=2, label=r'obs_ant')
        line.append( line_temp )
        line_temp, = p_position0.plot(t_stamp, des_ago[0,:], linewidth=2, label=r'des_ago')
        line.append( line_temp )
        line_temp, = p_position0.plot(t_stamp, des_ant[0,:], linewidth=2, label=r'des_ant')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
            
        p_position1 = fig2.add_subplot(312)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressures of Dof_1')
        line = []
        line_temp, = p_position1.plot(t_stamp, obs_ago[1,:], linewidth=2, label=r'obs_ago')
        line.append( line_temp )
        line_temp, = p_position1.plot(t_stamp, obs_ant[1,:], linewidth=2, label=r'obs_ant')
        line.append( line_temp )
        line_temp, = p_position1.plot(t_stamp, des_ago[1,:], linewidth=2, label=r'des_ago')
        line.append( line_temp )
        line_temp, = p_position1.plot(t_stamp, des_ant[1,:], linewidth=2, label=r'des_ant')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)
        
        p_position2 = fig2.add_subplot(313)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressures of Dof_2')
        line = []
        line_temp, = p_position2.plot(t_stamp, obs_ago[2,:], linewidth=2, label=r'obs_ago')
        line.append( line_temp )
        line_temp, = p_position2.plot(t_stamp, obs_ant[2,:], linewidth=2, label=r'obs_ant')
        line.append( line_temp )
        line_temp, = p_position2.plot(t_stamp, des_ago[2,:], linewidth=2, label=r'des_ago')
        line.append( line_temp )
        line_temp, = p_position2.plot(t_stamp, des_ant[2,:], linewidth=2, label=r'des_ant')
        line.append( line_temp )
        plt.legend(handles=line, loc=legend_position, shadow=True)

        plt.suptitle('Pressures Monitoring'+' Iter '+str(i_iter+1))
        plots.append(wandb.Image(plt, caption="matplotlib image"))

        ### to see the learned disturbances in ILC
        if disturbance is not None:
            fig3 = plt.figure(figsize=(18, 18))

            disturb0 = fig3.add_subplot(311)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Disturbance of Dof_0')
            line = []
            line_temp, = disturb0.plot(t_stamp, disturbance[0,:], linewidth=2, label=r'disturbance')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)       

            disturb1 = fig3.add_subplot(312)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Disturbance of Dof_1')
            line = []
            line_temp, = disturb1.plot(t_stamp, disturbance[1,:], linewidth=2, label=r'disturbance')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)      

            disturb2 = fig3.add_subplot(313)
            plt.xlabel(r'Time $t$ in s')
            plt.ylabel(r'Disturbance of Dof_2')
            line = []
            line_temp, = disturb2.plot(t_stamp, disturbance[2,:], linewidth=2, label=r'disturbance')
            line.append( line_temp )
            plt.legend(handles=line, loc=legend_position, shadow=True)

            plt.suptitle('Disturbance Visualization'+' Iter '+str(i_iter+1))
            plots.append(wandb.Image(plt, caption="matplotlib image"))                 
        
        wandb.log({'related plots': plots})
        plt.close()