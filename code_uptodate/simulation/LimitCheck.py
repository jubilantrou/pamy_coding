import numpy as np
import PAMY_CONFIG
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

anchor_ago_list = np.array([20500, 22000, 13850, 17000])
anchor_ant_list = np.array([20500, 20000, 13850, 17000])

ago_min_list = np.array([15000, 17000, 9850, 13000])
ago_max_list = np.array([26000, 27000, 17850, 19000])
ant_min_list = np.array([15000, 15000, 9850, 13000])
ant_max_list = np.array([26000, 25000, 17850, 21900])

limit_max = ago_max_list - anchor_ago_list
limit_min = ago_min_list - anchor_ago_list

def LimitCheck(u, dof):

    u[u>limit_max[dof]] = limit_max[dof]
    u[u<limit_min[dof]] = limit_min[dof]
    return u

if __name__ == '__main__':
    u = np.array([[7000], [-8000], [9000]])
    print(LimitCheck(u, 0))