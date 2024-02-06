'''
This script is the configuration of Pamy.
initial posture: GLOBAL_INITIAL
SI: identification data
pressure: anchor points and limits
strategy list
pid list
delta lists
weight list
'''
#%%
import numpy as np
from RealRobot import Robot
from RealRobotGeometry import RobotGeometry
import pickle5 as pickle
# %%
'''
The initial posture of pamy.
All the values are absolute values in angular space in rad.
'''
# GLOBAL_INITIAL = np.array([0.000000, 0.514884, 0.513349, 0.0000])
GLOBAL_INITIAL = np.array([0, 0.8, 0.8, 0])
# %%
def FulfillZeros( a ):
    b = np.zeros(  [len(a), len(max(a, key = lambda x: len(x)))]  )
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b
# %%
'''
All the necessary parameters that used to built the model and
inverse model of Pamy.
'''
dof_list = [0, 1, 2, 3]

# ----------------------------------------------------------------------------------------------------------------
'''
identification data: 01.06.2021
identification mode: ff
frequency range:     [0, 10]
'''
# inverse_model_num = [[ -1.603053083740808,  4.247255623123213,  -3.894718257376192, 1.248397916673204],
#                      [ -39.4562899165285, 107.6338556336952,   -101.4865793303678, 33.2833280994107],
#                      [-0.5340642923467604, 0.9405453623185243, -0.4103846610378554                   ],
#                      [  1.139061120808877, -1.134002042583525                                        ]]
# inverse_model_num = FulfillZeros( inverse_model_num ) * 1e5

# inverse_model_den = [[1.000000000000000, -2.397342550890251, 2.136262918863130, -0.683005338583667],
#                      [1.000000000000000, -1.972088694237271, 1.604586816873790, -0.428252150060600],
#                      [1.000000000000000, -1.702657081211864, 0.823186864989291                    ],
#                      [1.000000000000000, -0.825587854345462                                       ]]
# inverse_model_den = FulfillZeros( inverse_model_den )

# ndelay_list = np.array([2, 2, 3, 1])

# order_num_list = np.array([3, 3, 2, 1])
# order_den_list = np.array([3, 3, 2, 1])


# model_num = [[ -0.62380966054252,    1.49548544287500,  -1.33262144624559,    0.42606532841061],
#              [-0.0253445015260062, 0.0499816049205160, -0.0406674530288672, 0.0108538372707263],
#              [ -1.87243373940214,    3.18811256549306,  -1.54136285983862                     ],
#              [ 0.877916014980719,  -0.724796799103451                                         ]]
# model_num = FulfillZeros( model_num ) * 1e-5

# model_den = [[1.000000000000000, -2.649479088497819, 2.429562874042614, -0.778762680621906],
#              [1.000000000000000, -2.727926418358119, 2.572126764707656, -0.843549359806079],
#              [1.000000000000000, -1.761108869843411, 0.768418085460389                    ],
#              [1.000000000000000, -0.995558554204924                                       ]]
# model_den = FulfillZeros( model_den )   

# model_num_order   = [3, 3, 2, 1]
# model_den_order   = [3, 3, 2, 1]
# model_ndelay_list = [2, 2, 3, 1]

# anchor_ago_list = np.array([17500, 18500, 16000, 15000])
# anchor_ant_list = np.array([17500, 18500, 16000, 15000])
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
'''
!!!!! Last Used Parameters !!!!!
identification data: 10.10.2022
identification mode: ff
frequency range:     [0, 4]
'''
# inverse_model_num = [[-2.350995487550098, 6.644679608595308, -6.280551126507012, 1.986059415820528],
#                      [-24.62326418771043, 70.24241096821765, -66.78541829591975, 21.16608122268048],
#                      [-1.211340591489780, 3.525020609214648, -3.436926835229279, 1.122998695513753],
#                      [  1.139061120808877, -1.134002042583525                                     ]]
# inverse_model_num = FulfillZeros(inverse_model_num) * 1e5

# inverse_model_den = [[1.000000000000000, -1.800268767595627, 0.993209275691485, -0.171528458110958],
#                      [1.000000000000000, -1.347801996829548, 0.364428171765932, -0.001447535556446],
#                      [1.000000000000000, -2.222189096460812, 1.554090039489951, -0.320764252850695],
#                      [1.000000000000000, -0.825587854345462                                       ]]
# inverse_model_den = FulfillZeros(inverse_model_den)

# ndelay_list = np.array([0, 0, 0, 1])
# order_num_list = np.array([3, 3, 3, 1])
# order_den_list = np.array([3, 3, 3, 1])

# model_num = [[-0.268873573843934, 0.385131637331576, -0.125365716011132],
#              [-0.0256716800018761, 0.0251562582754755, -0.0001010131728973],
#              [-0.521835529387435, 0.967644660616160, -0.455002821472975],
#              [0.877916014980719, -0.724796799103451]]
# model_num = FulfillZeros(model_num) * 1e-5

# model_den = [[1.000000000000000, -2.826325972883738, 2.671443292752461, -0.844773810216940],
#              [1.000000000000000, -2.852684779432124, 2.712289393753596, -0.859596886153078],
#              [1.000000000000000, -2.910016087943825, 2.837291889147658, -0.927070968646912],
#              [1.000000000000000, -0.995558554204924]]
# model_den = FulfillZeros(model_den)   

# model_ndelay_list = [0, 0, 0, 1]
# model_num_order   = [2, 2, 2, 1]
# model_den_order   = [3, 3, 3, 1]

# anchor_ago_list = np.array([17500, 20700, 16000, 15000])
# anchor_ant_list = np.array([17500, 16300, 16000, 15000])
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
'''
identification data: 20.10.2022
identification mode: fb
frequency range:     [0, 4]
'''
# inverse_model_num = [[0.10563726446696005, -0.19816335192606250, 0.09330799511687839],
#                      [  0.696188563387390,   -1.327499022596144,   0.632364488636068],
#                      [0.09469560619999285, -0.09452160729660655],
#                      [  1.139061120808877, -1.134002042583525]]
# inverse_model_num = FulfillZeros(inverse_model_num) * 1e2

# inverse_model_den = [[1.000000000000000, -1.521654172804622, 0.727307919705600],
#                      [1.000000000000000, -0.368462983946576, 0.000214673390016],
#                      [1.000000000000000, -0.690319276955498],
#                      [1.000000000000000, -0.825587854345462]]
# inverse_model_den = FulfillZeros(inverse_model_den)

# ndelay_list = np.array([0, 0, 3, 1])
# order_num_list = np.array([2, 2, 1, 1])
# order_den_list = np.array([2, 2, 1, 1])

# model_num = [[0.094663564514468, -0.144045207956000, 0.068849560178935],
#              [0.009079732016178, -0.000005298412018],
#              [0.105601520506458, -0.072898765281419],
#              [0.877916014980719, -0.724796799103451]]
# model_num = FulfillZeros(model_num)

# model_den = [[1.000000000000000, -1.875884924945606, 0.883286741546229],
#              [1.000000000000000, -1.906809580635794, 0.908323580552979],
#              [1.000000000000000, -0.998162545123595],
#              [1.000000000000000, -0.995558554204924]]
# model_den = FulfillZeros(model_den)   

# model_ndelay_list = [0, 0, 3, 1]
# model_num_order   = [2, 1, 1, 1]
# model_den_order   = [2, 2, 1, 1]

# anchor_ago_list = np.array([17500, 20700, 16000, 15000])
# anchor_ant_list = np.array([17500, 16300, 16000, 15000])
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
'''
identification data: 04.12.2023
identification mode: ff
frequency range:     [0, 10]
obj:                 sim
'''
inverse_model_num = [[-686139.776350245,	1818668.61077509,	-1607353.20010400,	473739.600198047],
                     [-595810.253948733,	1657643.01045506,	-1540628.26910360,	478513.450732304],
                     [-90221.1003158467,	167340.240134087,	-77291.3125550815],
                     [-1655273.08363206,	4619075.13587724,	-4305785.35873620,	1341814.86184349]]
inverse_model_num = FulfillZeros(inverse_model_num)

inverse_model_den = [[1,	-0.793108906364425,	0.572350454862304,	-0.153007415376520],
                     [1,	-1.60869551791080,	0.820295911162373,	-0.0201266174981055],
                     [1,	-0.671571488472716,	4.40709013652154e-06],
                     [1,	-1.39789833851383,	0.983293829460442,	-0.222335685278340]]
inverse_model_den = FulfillZeros(inverse_model_den)

ndelay_list = np.array([2, 3, 3, 3])
order_num_list = np.array([3, 3, 2, 3])
order_den_list = np.array([3, 3, 2, 3])

model_num = [[-9.21270826464792e-07,	3.91751500835520e-07,	-3.83172453373049e-07],
             [-1.67838669001163e-06,	2.70001314554285e-06,	-1.37677373916589e-06,	3.37802469237754e-08],
             [-1.10838816695783e-05,	7.44361891089417e-06,	-4.88476655803705e-11],
             [-3.81882944318490e-07,	3.93346649220794e-07,	-2.30798997222552e-07]]
model_num = FulfillZeros(model_num)

model_den = [[1,	-2.65058035324677,	2.34260314808439,	-0.690441826180068],
             [1,	-2.78216596553858,	2.58576998111913,	-0.803130606700633],
             [1,	-1.85477942020504,	0.856687762446917],
             [1,	-2.79052150460992,	2.60125377577474,	-0.810630508712935]]
model_den = FulfillZeros(model_den)   

model_ndelay_list = [2, 3, 3, 3]
model_num_order   = [2, 3, 2, 2]
model_den_order   = [3, 3, 2, 3]
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
'''
identification data: 04.12.2023
identification mode: ff
frequency range:     [0, 10]
obj:                 real
'''
# inverse_model_num = [[-1.957403439384016e05,     3.824768202060075e05,    -1.881538753796024e05],
#                      [ 0.720071938819735e17,    -1.406797868487910e17,     0.689485753258883e17],
#                      [ 0.458297541661016e08,    -1.244466547878559e08,     1.140090083613778e08,    -0.352040169678436e08],
#                      [ 0.769687522482644e06,    -1.301751555631889e06,     0.556263126354410e06]]
# inverse_model_num = FulfillZeros(inverse_model_num)

# inverse_model_den = [[1,    -1.478182157401806,       0.831979388267108],
#                      [1,    -7.253621425499015e10,    2.668458196468245e10],
#                      [1,    -141.4704783208232,       103.6821551641323,    -19.0463731164319],
#                      [1,	-12.855754005778422,      4.594032316246684]]
# inverse_model_den = FulfillZeros(inverse_model_den)

# ndelay_list = np.array([4, 4, 2, 1])
# order_num_list = np.array([2, 2, 3, 2])
# order_den_list = np.array([2, 2, 3, 2])

# model_num = [[-0.510880884277333e-05,     0.755175007696411e-05,    -0.425042365578415e-05],
#              [ 0.000000000008779e-06,    -0.636764603896313e-06],
#              [ 0.004359355708581e-06,    -0.609153346871119e-06,    -0.613512702579700e-06],
#              [ 0.008212690739609e-04,    -0.102559051794143e-04]]
# model_num = FulfillZeros(model_num)

# model_den = [[1,    -1.954000961224277,    0.961242182341385],
#              [1,    -1.953690725393054,    0.957523430768619],
#              [1,    -2.715411789834645,    2.487663537276958,    -0.768147628290848],
#              [1,    -1.691272779676954,    0.722712932334113]]
# model_den = FulfillZeros(model_den)   

# model_ndelay_list = [4, 4, 2, 1]
# model_num_order   = [2, 1, 2, 1]
# model_den_order   = [2, 2, 3, 2]
# ----------------------------------------------------------------------------------------------------------------

anchor_ago_list = np.array([19300, 17500, 17100, 16000])
anchor_ant_list = np.array([19900, 22000, 16900, 17400])

ago_min_list = np.array([15000, 15000, 13000, 13000])
ago_max_list = np.array([24900, 21400, 20900, 18900])
ant_min_list = np.array([16000, 16000, 13000, 13000])
ant_max_list = np.array([24900, 24900, 20900, 21800])

# for real robot
# anchor_ago_list = np.array([20000, 19000, 17000, 17000])
# anchor_ant_list = np.array([20000, 20000, 17000, 17000])

# ago_min_list = np.array([15000, 15000, 13000, 13000])
# ago_max_list = np.array([25000, 21500, 21000, 21800])
# ant_min_list = np.array([16000, 16000, 13000, 13000])
# ant_max_list = np.array([25000, 25000, 21000, 19000])

ago_pressure_max = ago_max_list - anchor_ago_list
ago_pressure_min = ago_min_list - anchor_ago_list
ant_pressure_max = ant_max_list - anchor_ant_list
ant_pressure_min = ant_min_list - anchor_ant_list
pressure_max = [min([ago_pressure_max[i],-ant_pressure_min[i]]) for i in range(len(dof_list))]
pressure_min = [max([ago_pressure_min[i],-ant_pressure_max[i]]) for i in range(len(dof_list))]
pressure_limit = [min([ago_pressure_max[i],-ago_pressure_min[i],ant_pressure_max[i],-ant_pressure_min[i]]) for i in range(len(dof_list))]
# print('pressure limits:')
# print(pressure_limit)

strategy_list = np.array([1, 1, 1, 1])

# pid_list = [[-3.505924158687806e+04, -3.484022215671791e+05/5, -5.665386729745434e+02],
#             [-8.228984656729296e+04, -1.304087541343074e+04/2, -4.841489121599795e+02],
#             [-36752.24956301624,     -246064.5612272051/10,    -531.2866756516057],
#             [3.422187330173758e+04,  1.673663594798479e+05/10, 73.238165769446297]]

# pid_list = [[-3998,  -15108, -264.5],
#             [-3998,  -10108, -264.5],
#             [-3998,  -8008, -264.5],
#             [0, 0, 0]] # from simulink

pid_list = [[-8368.8*0.8,  -66950.4*0.8*0.8, -261.525],
            [-12000*0.6,  -51200*0.5, -703.125*0.75],
            [-16500,  -195200*0.2, -429*1.75],
            [-13800*0.85, -50468*0.5, -943]] # tuned PID

# pid_list = [[-5165, 0, 0],
#             [-9450, 0, 0],
#             [-14150, 0, 0],
#             [-20700, 0, 0]] # for parameters finding

# pid_list = [[-3615.5, -18077.5, -271.16],
#             [-5670,  -15019.84, -535.11],
#             [-4669.5,  -21066.99, -689.99],
#             [-4140, -44000, -150.25]] # tuned PID for real robot

# pid_list = np.array([[0.3*-10000,  0, 0.3*-260],
#             [0.25*-10500,  0, 0.25*-525],
#             [0.09*-18000,  0, 0.09*-875],
#             [-12000, 0, -1000]]) # tuned PD

pid_list = FulfillZeros( pid_list )

# delta_d_list   = np.array([1e-9, 1e-11, 1e-8, 1e-9]) 
# delta_y_list   = np.array([1e-3, 1e-3, 1e-3, 1e-3])
# delta_w_list   = np.array([1e-9, 1e-11, 1e-8, 1e-9])
# delta_ini_list = np.array([1e-7, 1e-7, 1e-7, 1e-7])
delta_d_list = np.zeros(4)
delta_y_list = np.zeros(4)
delta_w_list = np.zeros(4)
delta_ini_list = np.zeros(4)

# weight_list = [(0.1, 1.0),
#                (1.0, 1.0),
#                (0.1, 1.0),
#                (1.0, 1.0)]
weight_list = [(0.0, 0.0),
               (0.0, 0.0),
               (0.0, 0.0),
               (0.0, 0.0)]
# %%
'''
Build the robot model
'''
def build_pamy(frontend=None):
    Pamy = Robot(frontend, dof_list, model_num, model_den,
                 model_num_order, model_den_order, model_ndelay_list,
                 inverse_model_num, inverse_model_den, order_num_list, order_den_list,
                 ndelay_list, anchor_ago_list, anchor_ant_list, strategy_list, pid_list,
                 delta_d_list, delta_y_list, delta_w_list, delta_ini_list,
                 pressure_min, pressure_max, weight_list)
    return Pamy
# %%
def build_geometry():
    Pamy_geometry = RobotGeometry(initial_posture=GLOBAL_INITIAL)
    return Pamy_geometry
# %%
def get_trajectory(index, mode='train'):
    path_file = "/home/hao/Desktop/Learning/data/online_optimization" + '/' + mode + '/' + str(index)
    file = open(path_file, 'rb')
    t = pickle.load(file)
    p = pickle.load(file)
    file.close()
    return (t, p)
# %%
# def get_recorded_ball(Geometry=None):
#     path_of_file = "/home/hao/Desktop/Learning/data/" + 'BallsData' + '.txt'
#     file = open(path_of_file, 'rb')
#     time_list = pickle.load(file)
#     position_list = pickle.load(file)
#     velocity_list = pickle.load(file)
#     file.close()
#     position_mean = np.mean(position_list, axis=0)
#     # offset angles of upright posture
#     offset = np.array( [2.94397627, -0.078539855235, -0.06333859293225] )
#     # angles of initial posture
#     angle_initial_ref = np.array( [2.94397627, -0.605516948, -0.5890489142699] )
#     # anchor angles
#     angle_anchor_ref = np.array( [2.94397627, -1.452987321865, -0.87660612618] )
#     angle_initial = angle_initial_ref - offset
#     angle_anchor = angle_anchor_ref - offset
#     (_, position_anchor) = Geometry.AngleToEnd(angle_anchor, frame='Cartesian')
#     position_error = position_anchor - position_mean
#     position_list = position_list + position_error  # have been calibrated
#     return (time_list, position_list)