import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
import math

dof = 0
index = 7

path_of_file = "/home/hao/Desktop/Hao/data_single_joint/" + 'single_play_' + str(dof) + '_' + str(index) + '.txt'
# path_of_file = "/home/hao/Desktop/Hao/data/" + "F_SmoothSignal_1_a__i_40.txt"
file = open(path_of_file, 'rb')
t_stamp_u = pickle.load(file) # time stamp for x-axis
y_history = pickle.load(file) 
ff_history = pickle.load(file) 
disturbance_history = pickle.load(file) 
P_history = pickle.load(file)
time_history = pickle.load(file) 
file.close()

Location = 'lower left'

print("begin to draw the plot")

Nr = len( y_history )
# row_list = [40]
row_list = np.arange(38, 41)

line = []
plt.figure('dof' + str(dof))
plt.xlabel(r'Time $t$ in s')
plt.ylabel(r'Angle $\theta$ in degree')
line_temp, = plt.plot(t_stamp_u, (y_history[0] - y_history[0][0]) * 180 / math.pi, label=r'Desired Trajectory', linewidth=1)
line.append(line_temp)
for irow in row_list:
    line_temp, = plt.plot(t_stamp_u, ( y_history[irow+1] - y_history[irow+1][0] ) * 180 / math.pi, label=r'Iteration {}'.format(irow), linewidth=0.3)
    line.append(line_temp)
plt.legend(handles = line, loc=Location, shadow=True)
plt.show()


line = []
plt.figure('dof' + str(dof))
plt.xlabel(r'Time $t$ in s')
plt.ylabel(r'Pressure $p$ in kPa')
for irow in row_list:
    line_temp, = plt.plot(t_stamp_u, ff_history[irow], label=r'Iteration {}'.format(irow), linewidth=0.3)
    line.append(line_temp)
plt.legend(handles = line, loc=Location, shadow=True)
plt.show()


line = []
plt.figure('dof' + str(dof))
plt.xlabel(r'Time $t$ in s')
plt.ylabel(r'Disturbance')
for irow in row_list:
    line_temp, = plt.plot(t_stamp_u, disturbance_history[irow], label=r'Iteration {}'.format(irow), linewidth=0.3)
    line.append(line_temp)
plt.legend(handles = line, loc=Location, shadow=True)
plt.show()