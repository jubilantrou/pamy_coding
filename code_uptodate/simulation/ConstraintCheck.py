# %%
import math
import numpy as np
import time
import o80
import o80_pam
import matplotlib.pyplot as plt
from scipy import integrate
from pam_handle import get_handle
from RealJoint import Joint
import pickle5 as pickle
# %%
def FulfillZeros( a ):
    b = np.zeros(  [len(a), len(max(a, key = lambda x: len(x)))]  )
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b
# %%
fs = 100

inverse_model_num = [[-6.63039064416817, 11.84600864250952, -5.23279947917328],
                     [39.4562899165285, -107.6338556336952, 101.4865793303678, -33.2833280994107],
                     [-0.5340642923467604, 0.9405453623185243, -0.4103846610378554],
                     [1.139061120808877, -1.134002042583525]]
inverse_model_num = FulfillZeros( inverse_model_num ) * 1e5

inverse_model_den = [[1.000000000000000, -1.639490049424085, 0.846522120098606],
                     [1.000000000000000, -1.972088694237271, 1.604586816873790, -0.428252150060600],
                     [1.000000000000000, -1.702657081211864, 0.823186864989291],
                     [1.000000000000000, -0.825587854345462]]
inverse_model_den = FulfillZeros( inverse_model_den )

ndelay_list = np.array([2, 2, 3, 1])

order_num_list = np.array([2, 3, 2, 1])
order_den_list = np.array([2, 3, 2, 1])

model_num = [[-0.150820676136113, 0.247268997772569, -0.127673038517447],
             [],
             [],
             []]
model_num = FulfillZeros( model_num ) * 1e-5
model_den = [[1.000000000000000, -1.786623032977520, 0.789214355533612],
             [],
             [],
             []]
model_den = FulfillZeros( model_den )   

model_num_order = [2, 2, 2, 2]
model_den_order = [2, 2, 2, 2]
model_ndelay_list = [2, 2, 2, 2]

pressure_list = [[],
                 [],
                 [21500, 21000, 20500, 20000, 19500, 19000, 18500, 18000, 17500, 17000, 16500, 16000, 15500, 15000, 14500],
                 []]
pressure_list = FulfillZeros( pressure_list )           

section_list = [[],
                [],
                [-1.0664, -1.0140, -0.9540, -0.8829, -0.7942, -0.6714, -0.4495, 0.0637, 0.4918, 0.6981, 0.8115, 0.8956, 0.9639, 1.0220, 1.0730],
                []]
section_list = FulfillZeros( section_list )

anchor_ago_list = np.array([17500, 18500, 16000, 15000])
anchor_ant_list = np.array([17500, 18500, 16000, 15000])

strategy_list = np.array([1, 2, 2, 2])

pid_list = [[-54105.12230072278, -138908.2295581382, -1875.513442094147],
            [8.228984656729296e+04, 1.304087541343074e+04, 4.841489121599795e+02],
            [-10751.42381499189, -72222.74350177329, -244.4645553027873],
            [3.422187330173758e+04, 1.673663594798479e+05, 73.238165769446297]]

# pid_list = np.array([[-13000, 0, -300],
#                      [80000, 0, 300],
#                      [-5000, -8000, -100],
#                      [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])

pid_list = FulfillZeros( pid_list )

strategy_list = np.array([2, 2, 2, 2])
# %%
T = 5
t = 1 / fs
Ns = round(T/t) + 1
t_stamp_u = np.linspace(0, T, Ns, endpoint=True)

# connecting to the shared memory of the robot's controller
frontend = o80_pam.FrontEnd("real_robot")
# %% define the second joint
sgn = -1
dof = 2
'''
frontend, dof, anchor_ago, anchor_ant,
num, den, order_num, order_den, ndelay,
pid, strategy
'''
Joint = Joint(frontend, dof, anchor_ago_list[dof], anchor_ant_list[dof], 
                inverse_model_num[dof], inverse_model_den[dof],
                order_num_list[dof], order_den_list[dof], ndelay_list[dof],
                pid_list[dof], strategy_list[dof])
# %% 
Joint.PressureInitialization( anchor_ago_list[dof], anchor_ant_list[dof] )
theta = frontend.latest().get_positions()
angle_initial = theta[dof]
Joint.AngleInitialization( angle_initial )
'''
y, mode_name="fb+ff", mode_trajectory="ref",
frequency_frontend=100, frequency_backend=500,
ifplot="yes", u_ago=[], u_ant=[], ff=[], echo="no"
'''
u_ago = np.zeros( len(t_stamp_u) ) + anchor_ago_list[dof]
u_ant = np.zeros( len(t_stamp_u) ) + anchor_ant_list[dof]
ff = np.zeros( len(t_stamp_u) )

ff_list = []
y_list = []
obs_pressure_list = []
for amp in range( 10 ) :
    ff[101:402] += sgn * 1000
    (y_out, _, obs_pressure) = Joint.Control(mode_name='ff', ifplot='no',
                              u_ago=u_ago, u_ant=u_ant, ff=ff, echo='yes')
    ff_list.append( np.copy( ff ) )
    y_list.append( np.copy( y_out ) )
    obs_pressure_list.append( np.copy( obs_pressure) )
    Joint.AngleInitialization( angle_initial, T=5 ) 
    Joint.PressureInitialization( anchor_ago_list[dof], anchor_ant_list[dof] )
# %% save the data
path_of_file = "/home/hao/Desktop/Hao/ConstraintCheck/" + 'dof_' + str(dof) + '_sgn_' + str(sgn)
file = open(path_of_file, 'wb')
pickle.dump(t_stamp_u, file, -1) # time stamp for x-axis
pickle.dump(ff_list, file, -1)
pickle.dump(y_list, file, -1)
pickle.dump(obs_pressure_list, file, -1)
file.close()