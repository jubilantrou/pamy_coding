'''
This script is used to define the function 
that limits the computed inputs within the allowed ranges.
'''
import numpy as np

# TODO: need to solve the prolem of a circular import so that we can import PAMY_CONFIG directly for desired values below
anchor_ago_list = np.array([20500, 16000, 13850, 17000])
anchor_ant_list = np.array([20500, 15500, 13850, 17000])
ago_min_list = np.array([12000, 10000, 8850,  13000])
ago_max_list = np.array([29000, 22000, 18850, 19000])
ant_min_list = np.array([12000, 9500, 8850,  13000])
ant_max_list = np.array([29000, 21500, 18850, 21900])

ago_pressure_max = ago_max_list - anchor_ago_list
ago_pressure_min = ago_min_list - anchor_ago_list
ant_pressure_max = ant_max_list - anchor_ant_list
ant_pressure_min = ant_min_list - anchor_ant_list
limit_max = [min([ago_pressure_max[i],-ant_pressure_min[i]]) for i in range(4)]
limit_min = [max([ago_pressure_min[i],-ant_pressure_max[i]]) for i in range(4)]

def LimitCheck(u, dof):
    u[u>limit_max[dof]] = limit_max[dof]
    u[u<limit_min[dof]] = limit_min[dof]
    return u

if __name__ == '__main__':
    u = np.array([[9000], [-9000], [9000]])
    print(LimitCheck(u, 0))