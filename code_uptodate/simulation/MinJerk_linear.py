import numpy as np
import math
import matplotlib.pyplot as plt

def CalPath(x_0, slope, T, step):
    nr_dof = len(x_0)
    nr_point = int( np.round(T/step) ) + 1
    p = np.zeros((nr_dof, nr_point))
    
    for i_point in range( nr_point ):
        p[:, i_point] = x_0 + i_point*step*slope
    
    return p

def PathPlanning(x_list, t_list, delta_list, step):
    
    nr_dof = x_list.shape[0]
    nr_point = x_list.shape[1]
    slope = np.zeros(nr_dof)
    t_stamp = np.linspace(t_list[0], t_list[-1], int(t_list[-1]/step)+1, endpoint=True)
    
    position = np.zeros((nr_dof, len(t_stamp)))
    velocity = np.zeros((nr_dof, len(t_stamp)))
    acceleration = np.zeros((nr_dof, len(t_stamp)))
    jerk = np.zeros((nr_dof, len(t_stamp)))
    
    position[:, 0] = x_list[:, 0]
    velocity[:, 0] = np.zeros(nr_dof)
    acceleration[:, 0] = np.zeros(nr_dof)
    jerk[:, 0] = np.zeros(nr_dof)
    
    idx_1 = 0
    idx_2 = 1
    
    for i_point in range(1, nr_point):
        
        if delta_list[i_point-1] > 0:
            idx_1 = idx_2
            idx_2 = int((t_list[i_point-1]+delta_list[i_point-1])/step) + 1
            p_temp = CalPath( position[:, idx_1-1], 
                              slope, 
                              delta_list[i_point-1],
                              step )
            position[:, idx_1:idx_2] = p_temp[:, 1:]
        
        idx_1 = idx_2
        idx_2 = int(t_list[i_point]/step) + 1
        slope = (x_list[:, i_point] - position[:, idx_1-1]) / (t_list[i_point]-(t_list[i_point-1]+delta_list[i_point-1]))
        p_temp = CalPath( position[:, idx_1-1], 
                          slope, 
                          t_list[i_point]-(t_list[i_point-1]+delta_list[i_point-1]),
                          step )
        
        position[:, idx_1:idx_2] = p_temp[:, 1:]
    
    for i_point in range(1, len(t_stamp) ):
        velocity[:, i_point] = (position[:, i_point]-position[:, i_point-1])/step
        acceleration[:, i_point] = (velocity[:, i_point]-velocity[:, i_point-1])/step
        jerk[:, i_point] = (acceleration[:, i_point]-acceleration[:, i_point-1])/step

    return (position, velocity, acceleration, jerk, t_stamp)

if __name__ == '__main__':
  # step size 
  step = 0.01
  '''
  position/velocity/acceleration/duration for all dofs at different time points
  dimension = nr_dof * nr_point 
  '''
  x_list = [[0.0, math.pi/4, 0.0, 0.0],
            [0.0, math.pi/3, 0.0, 0.0],
            [0.0, math.pi/2, 0.0, 0.0]]
  x_list = np.array(x_list)
  t_list = np.array([0.0, 1.0, 2.0, 2.2])
  delta_list = np.array([0.1, 0.1, 0.0])
    
  [position, velocity, acceleration, jerk, t_stamp] = PathPlanning(x_list, t_list, delta_list, step)
  
  nr_dof = x_list.shape[0]
  legend_position = 'lower right'
  
  fig = plt.figure(figsize=(16, 16))
  ax_position = fig.add_subplot(411)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Angle $\theta$ in degree')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_position.plot(t_stamp, position[i, :] * 180 / math.pi, linewidth=2, label=r'Pos. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
      
  ax_velocity = fig.add_subplot(412)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Velocity $v$ in rad/s')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_velocity.plot(t_stamp, velocity[i, :], linewidth=2, label=r'Vel. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
  
  ax_acceleration = fig.add_subplot(413)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Acceleration $a$ in rad/$s^2$')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_acceleration.plot(t_stamp, acceleration[i, :], linewidth=2, label=r'Acc. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
  
  ax_jerk = fig.add_subplot(414)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Jerk $j$ in rad/$s^3$')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_jerk.plot(t_stamp, jerk[i, :], linewidth=2, label=r'Jerk. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
  
  plt.show()