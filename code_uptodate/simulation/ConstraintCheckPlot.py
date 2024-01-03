# %%
import pickle5 as pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import RealTrajectoryGeneration as RTG
# %% read data
Location = 'lower right'

dof_list = [1]

number_to_show = 5

path_of_folder = "/home/hao/Desktop/Hao/ConstraintCheck"

files = os.listdir( path_of_folder )

for file in files:

    if not os.path.isdir( file ):
        path_of_file = path_of_folder + '/' + file
        # open file and read the data
        f = open(path_of_file, 'rb')
        t_stamp_u = pickle.load( f ) # time stamp for x-axis
        ff_list = pickle.load( f ) 
        y_list = pickle.load( f )
        obs_pressure_list = pickle.load( f )
        f.close()
        
        num = len( ff_list )

        line = []
        plt.figure( figsize=(16, 8) )
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Pressure')

        for i in range(num):
            line_temp, = plt.plot(t_stamp_u, ff_list[i],
                    label=r'Nr {}'.format(i+1),
                    linestyle='--', linewidth=2)
            line.append(line_temp)
            line_temp, = plt.plot(t_stamp_u, obs_pressure_list[i],
                    label=r'Nr {}'.format(i+1),
                    linewidth=1)
            line.append(line_temp)
        plt.legend(handles=line, loc=Location, shadow=True)
        
        # plt.show()
        plt.savefig(path_of_folder+ '/' + file + '_ff' + '.pdf')

        line = []
        plt.figure( figsize=(16, 8) )
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Angle $\theta$ in degree')

        for i in range(num):
            line_temp, = plt.plot(t_stamp_u, (y_list[i] - y_list[i][0]) * 180 / math.pi,
                    label=r'Nr {}'.format(i+1),
                    linewidth=1)
            line.append(line_temp)
        plt.axvline( 1, linestyle='--' )
        plt.axvline( 4, linestyle='--' )
        plt.legend(handles=line, loc=Location, shadow=True)
        # plt.show()
        plt.savefig(path_of_folder+ '/' + file + '_path' + '.pdf')
    