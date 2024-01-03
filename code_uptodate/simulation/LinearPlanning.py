import numpy as np
import math
import matplotlib.pyplot as plt

def GetPath(x, v, t, step):
    n = 0.3

    nr = t / step
    nr = nr.astype(np.int)
    p = np.ones(int(np.round((t[-1]-t[0])/step))+1) * x[2]

    p[0:11] = x[0]
    tx = (10*n*v[1]*step + nr[1]*v[1]*step + x[0]-x[1]) / ((n+1)*v[1]*step)
    for i in range(11, int(np.ceil(tx))):
        p[i] = x[0] + (-n*v[1]*step) * (i-11)
    for i in range(int(np.ceil(tx)), nr[1]+11):
        p[i] = x[1] + v[1]*(i-nr[1])*step
    
    a = (x[2]-p[nr[1]+10])/(t[2]-t[1]-0.1)
    for i in range(nr[1]+11, nr[2]+1):
        p[i] = x[2] + a*(i-nr[2])*step
    
    return p

def GetFlatPath(x, t, step):
    nr = t / step
    nr = nr.astype(np.int)
    p = np.ones(int(np.round((t[-1]-t[0])/step))+1) * x[2]
    p[nr[1]-10: nr[1]+11] = x[1]
    a = (x[1]-x[0]) / (t[1]-t[0]-0.2) * step
    for i in range(nr[0]+11, nr[1]-9):
        p[i] = a * (i - nr[0] - 11) + x[0]
    a = (x[2]-x[1]) / (t[2]-t[1]-0.1) * step
    for i in range(nr[1]+11, nr[2]):
        p[i] = a * (i - nr[2]) + x[2]
    
    return p

def PathPlanning(x_list, v_list, t_list, step):
    t_stamp = np.linspace(t_list[0], t_list[-1], int(np.round((t_list[-1]-t_list[0])/step))+1, endpoint=True)
    p = np.zeros((3, len(t_stamp)))
    p[0, :] = GetPath(x_list[0, :], v_list[0, :], t_list, step)
    for i_dof in range(1, 3):
        p[i_dof, :] = GetFlatPath(x_list[i_dof, :], t_list, step)
    
    return p, t_stamp

if __name__ == '__main__':
    step = 0.01

    x_list = [[0.0, math.pi/4, 0.0, 0.0],
            [0.0, math.pi/4, 0.0, 0.0],
            [0.0, math.pi/4, 0.0, 0.0]]
    x_list = np.array(x_list)
    v_list = [[0.0, 5.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0]]
    v_list = np.array(v_list)
    
    t_list = [0.0, 1.0, 2.0, 2.2]
    t_list = np.array(t_list)

    nr = t_list / step
    nr = nr.astype(np.int)

    [position, t_stamp] = PathPlanning(x_list, v_list, t_list, step)

    legend_position = 'lower right'
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    for i_dof in range(3):
        ax = axs[i_dof]
        ax.set_xlabel(r'Time $t$ in s')
        ax.set_ylabel(r'Angle $\theta$ in degree')
        line = []
        line_temp, = ax.plot(t_stamp, position[i_dof, :] * 180 / math.pi, linewidth=1.5, label=r'dof {}'.format(i_dof))
        line.append( line_temp )
        for i in range(len(t_list)):
            ax.axvline(t_stamp[nr[i]], color='red', linewidth=0.5, linestyle='--')
        ax.legend(handles=line, loc=legend_position, shadow=True)
    plt.show() 