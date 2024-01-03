import numpy as np
# %% add the limits
"""
max_pressure_ago = [22000, 25000, 22000, 22000]
max_pressure_ant = [22000, 23000, 22000, 22000]

min_pressure_ago = [13000, 13500, 10000, 8000]
min_pressure_ant = [13000, 14500, 10000, 8000]
"""

"""
anchor = [17500 18500 16000 15000]
"""
anchor_ago_list = np.array([17500, 20700, 16000, 15000])
anchor_ant_list = np.array([17500, 16300, 16000, 15000])

ago_max_list = np.array([22000, 25000, 22000, 22000])
ant_max_list = np.array([22000, 23000, 22000, 22000])

ago_min_list = np.array([13000, 13500, 10000, 8000])
ant_min_list = np.array([13000, 14500, 10000, 8000])

limit_max = ago_max_list - anchor_ago_list
limit_min = ago_min_list - anchor_ago_list

def LimitCheck(u, dof):

    u[u>limit_max[dof]] = limit_max[dof]
    u[u<limit_min[dof]] = limit_min[dof]
    return u

if __name__ == '__main__':
    u = np.array([[7000], [-8000], [9000]])
    print(LimitCheck(u, 0))