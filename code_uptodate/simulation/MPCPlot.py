# %%
import pickle5 as pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import os
# %% read data
Location = 'lower right'

dof_list = [0,1,2]


index = 11

path_of_folder = "/home/hao/Desktop/HitBall/test"

files = os.listdir( path_of_folder )

frequency = 50

for file in files:
    path_of_figure = path_of_folder + '/' + file

    path_of_file = path_of_folder + '/' + file + '/' + 'data'

    '''
    pickle.dump(trajectory_history, file, -1) 
    pickle.dump(trajectory_real, file, -1)
    pickle.dump(p_in_cylinder, file, -1)
    pickle.dump(v_in_cylinder, file, -1)
    pickle.dump(p_to_check, file, -1)
    pickle.dump(position_list[index], file, -1)
    pickle.dump(time_list[index], file, -1)
    '''
    
    # open file and read the data
    f = open(path_of_file, 'rb')
    y_history = pickle.load( f )
    y_real = pickle.load( f )
    y_for_NN = pickle.load( f )
    p_in_cylinder = pickle.load( f )
    v_in_cylinder = pickle.load( f )
    p_to_check = pickle.load( f )
    ff = pickle.load( f )
    fb = pickle.load( f )
    position_of_ball = pickle.load( f )
    hitting_time = pickle.load( f )
    f.close()

    number_to_show = len( y_history )

    for dof in dof_list:
        line = []
        plt.figure( figsize=(32, 8) )
        plt.subplot(4, 1, 1)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Angle $\theta$ in degree')

        l = y_history[0].shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        line_temp, = plt.plot(t_stamp,
                            ( y_history[0][dof, :] ) * 180 / math.pi,
                                label=r'First', linewidth=1.0)
        line.append( line_temp )

        for i in range(1, number_to_show ):
            l = y_history[i].shape[1]
            t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
            line_temp, = plt.plot(t_stamp,
                                    ( y_history[i][dof, :] ) * 180 / math.pi,
                                    linestyle='--', linewidth=0.5)
            line.append(line_temp) 

        l = y_for_NN.shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        line_temp, = plt.plot(t_stamp,
                            ( y_for_NN[dof, :] ) * 180 / math.pi,
                              label=r'For NN', linewidth=1.0)
        line.append(line_temp)

        l = y_real.shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        line_temp, = plt.plot(t_stamp,
                            ( y_real[dof, :] - y_real[dof, 0] ) * 180 / math.pi,
                            label=r'Real', linewidth=1.0)
        line.append(line_temp)
        plt.legend(handles = line, loc=Location, shadow=True)

        plt.subplot(4, 1, 2)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Position in Cylinder')
        l = p_in_cylinder.shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        plt.plot(t_stamp, p_in_cylinder[dof, :], linestyle='-', 
                                label=r'P in Cylinder', linewidth=1.0)

        l = p_to_check.shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        plt.plot(t_stamp, p_to_check[dof, :], linestyle='--', 
                                label=r'P to check', linewidth=2.0)

        plt.legend(ncol=2, loc=Location, shadow=True)

        plt.subplot(4, 1, 3)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Velocity in Cylinder')
        l = v_in_cylinder.shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        plt.plot(t_stamp, v_in_cylinder[dof, :], linestyle='-', 
                                label=r'V in Cylinder', linewidth=1.0)


        plt.legend(ncol=1, loc=Location, shadow=True)

        plt.subplot(4, 1, 4)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure')
        l = ff.shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        plt.plot(t_stamp, ff[dof, :], linestyle='-', 
                                label=r'Feedforward', linewidth=1.0)

        l = fb.shape[1]
        t_stamp = np.linspace(0, (l-1)/frequency, num=l, endpoint=True )
        plt.plot(t_stamp, fb[dof, :], linestyle='-', 
                                label=r'Feedback', linewidth=1.0)

        plt.legend(ncol=2, loc=Location, shadow=True)

        plt.show()
        # plt.savefig(path_of_figure + '/' + str(dof) + '.pdf')


# for dof in dof_list:
#     line = []
#     plt.figure( file )
#     plt.xlabel(r'Time $t$ in s')
#     plt.ylabel(r'Disturbance')
#     for irow in row_list:
#         line_temp, = plt.plot(t_stamp_u,
#                             disturbance_history[irow][dof, :],
#                             label=r'Iteration {}'.format(irow),
#                             linewidth=1)
#         line.append(line_temp)       
#     plt.legend(handles = line, loc=Location, shadow=True)
#     # plt.show()
#     plt.savefig(path_of_figure + '/' + file + '.pdf')

# for dof in dof_list:
#     line = []
#     plt.figure( file )
#     plt.xlabel(r'Time $t$ in s')
#     plt.ylabel(r'Pressure')
#     for irow in row_list:
#         line_temp, = plt.plot(t_stamp_u,
#                             ff_history[irow][dof, :],
#                             label=r'Iteration {}'.format(irow),
#                             linewidth=1)
#         line.append(line_temp)       
#     plt.legend(handles = line, loc=Location, shadow=True)
#     # plt.show()
#     plt.savefig(path_of_figure + '/' + file + '.pdf')