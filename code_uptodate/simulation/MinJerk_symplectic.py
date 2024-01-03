import math
from scipy.special import eval_legendre
import numpy as np
import scipy.linalg
import lemkelcp as lcp
import matplotlib.pyplot as plt

# %% Define global variables
# number of states
ns = 3
# number of inputs
nu = 1
# number of constraints
nh = 2

def LGL( n ):
    m = n - 2
    A = np.zeros([n-1, n-1])
    for iM in range(1, m+1):
        b = np.sqrt( iM * (iM + 2) / ((iM * 2 + 1) * (iM * 2 + 3)) )
        A[iM-1, iM] = b
        A[iM, iM-1] = b
    # calculate the eigenvalue of matrix A
    eig = np.linalg.eig( A )
    v = eig[0]
    v = np.sort( v )
    # the support points to calculate legendre polynomial
    lg = np.array([])
    lg = np.append(lg, -1)
    lg = np.append(lg, v)
    lg = np.append(lg, 1)

    leg0dot = eval_legendre(n, lg)

    w = np.array([])
    for iw in range(n + 1):
        w = np.append(w, 2/(n*(n+1)*leg0dot[iw]*leg0dot[iw]))

    return (w, leg0dot, lg)

def InitialSolution(x0, xf, t0, tf):
    global ns, nu, nh, time_length, number_interval, the_ends, the_degrees, cpt_number

    slope = ( xf - x0 ) / time_length
    x_initial = np.array([])
    lam_initial = np.array([])

    for iE in range(number_interval):
        time_left = the_ends[iE]
        time_right = the_ends[iE + 1]
        sublength = time_right - time_left
        (_, _, nodes_standard) = LGL(the_degrees[iE])

        nodes_real = sublength / 2 * nodes_standard + (time_left + time_right) / 2

        for iD in range(the_degrees[iE] + 1):
            x_temp = slope * (nodes_real[iD] - t0) + x0
            x_initial = np.append(x_initial, x_temp)
    
    u_initial = np.zeros([cpt_number * nu, 1])
    lam_initial = np.zeros([cpt_number * ns, 1])
    
    return (x_initial, u_initial, lam_initial)

def FirstOrderDerivativeMatrix(x, n):

    D = np.zeros([n+1, n+1])
    leg0dot = eval_legendre(n, x)
    for kN in range(n+1):
        for lN in range(n+1):
            if kN != lN:
                D[kN, lN] = leg0dot[kN] / (leg0dot[lN] * (x[kN] - x[lN]))

    D[0, 0] = -n * (n + 1) / 4
    D[n, n] = n * (n + 1) / 4

    return D

def CoefficientMatrix(x, u):
    global ns, nu, nh, constraint_max, constraint_min

    weight_P = 0
    weight_R = 1

    th = np.array([ x[0] - constraint_max, constraint_min - x[0]]).reshape(-1, 1)

    sub_A = np.array( [[0, 1, 0], [0, 0, 1], [0, 0, 0]] )
    sub_zero = np.zeros([3, 3])

    # l1 = np.hstack((np.hstack((sub_A, sub_zero)), sub_zero))
    # l2 = np.hstack((np.hstack((sub_zero, sub_A)), sub_zero))
    # l3 = np.hstack((np.hstack((sub_zero, sub_zero)), sub_A))
    # tA = np.vstack((np.vstack((l1, l2)), l3))

    tA = np.copy( sub_A )

    tB = np.zeros([ns, nu])
    tB[-1, 0] = 1

    tW = np.zeros([ns, 1])

    # tC = np.array([[2*x[0], 0, 0, 2*x[3], 0, 0, 2*x[6], 0, 0],
    #                [-2*x[0], 0, 0, -2*x[3], 0, 0, -2*x[6], 0, 0]])
    
    tC = np.array([[1, 0, 0],
                   [-1, 0, 0]])

    tD = np.zeros([nh, nu])

    tV = th - np.dot(tC, x).reshape(-1, 1) - np.dot(tD, u).reshape(-1, 1)
    tP = np.identity(ns) * weight_P
    tR = np.identity(nu) * weight_R

    return (tA, tB, tC, tD, tW, tV, tP, tR)

def CalSubMatrix(x, u, ii):
    global ns, nu, nh, number_interval, the_degrees, the_etas

    interval_length = the_etas[ii]
    degree = the_degrees[ii]

    Kxx = np.zeros([(degree+1)*ns, (degree+1)*ns])
    Kxlam = np.copy(Kxx)
    Klamlam = np.copy(Kxx)

    kesi_x = np.zeros([(degree+1)*ns, (degree+1)*nh])
    kesi_lam = np.copy(kesi_x)

    zeta_x = np.zeros([(degree+1)*ns, 1])
    zeta_lam = np.copy(zeta_x)

    v = np.zeros([(degree+1)*nh, 1])

    (wg, _, xg) = LGL(degree)

    D = FirstOrderDerivativeMatrix(xg, degree)

    for iN in range(degree + 1):
        index1 = iN * ns
        index2 = (iN + 1) * ns - 1

        index1_h = iN * nh 
        index2_h = (iN + 1) * nh - 1 

        index1_u = iN * nu
        index2_u = (iN + 1) * nu - 1 
        
        xi = x[index1 : index2 + 1]
        ui = u[index1_u : index2_u + 1]

        (tA, tB, tC, tD, tW, tV, tP, tR) = CoefficientMatrix(xi, ui)

        BinvRBT = np.matmul(np.matmul(tB, np.linalg.inv(tR)), tB.T)
        DinvRBT = np.matmul(np.matmul(tD, np.linalg.inv(tR)), tB.T)
        DinvRDT = np.matmul(np.matmul(tD, np.linalg.inv(tR)), tD.T)

        Kxx[index1:index2+1, index1:index2+1] = interval_length / 2 * wg[iN] * tP
        Klamlam[index1:index2+1, index1:index2+1] = -interval_length / 2 * wg[iN] * BinvRBT

        for jN in range(degree + 1):
            Kxlam[index1:index2+1, jN*ns:(jN+1)*ns] = -wg[jN] * D[jN, iN] * np.identity(ns)
        
        Kxlam[index1:index2+1, index1:index2+1] += interval_length / 2 * wg[iN] * tA.T
        kesi_x[index1:index2+1, index1_h:index2_h+1] = interval_length / 2 * wg[iN] * tC.T
        kesi_lam[index1:index2+1, index1_h:index2_h+1] = -interval_length / 2 * wg[iN] * DinvRBT.T
        zeta_lam[index1:index2+1] = interval_length / 2 * wg[iN] * tW

        v[index1_h : index2_h + 1] = np.copy( tV )
        if iN == 0:
            G = np.copy( tC )
            H = np.copy(DinvRBT)
            M = np.copy( DinvRDT )
        else:
            G = scipy.linalg.block_diag(G, tC)
            H = scipy.linalg.block_diag(H, DinvRBT)
            M = scipy.linalg.block_diag(M, DinvRDT)
    
    Kxlam[-1-ns+1:, -1-ns+1:] += np.identity(ns)
    K_1 = np.hstack((Kxx, Kxlam))
    K_2 = np.hstack((Kxlam.T, Klamlam))
    K = np.vstack((K_1, K_2))

    kesi = np.vstack((kesi_x, kesi_lam))

    zeta = np.vstack((np.zeros([(degree+1)*ns, 1]), zeta_lam))

    return (K, kesi, zeta, G, H, M, v)

def CalGlobalMatrix(x, u, lam):
    global ns, nu, nh, number_interval, the_degrees, the_etas, x0, xf, index12
    K_size = 0
    GHM_row_size = 0
    G_col_size = 0

    index1 = 0
    index2 = ns * (the_degrees[0] * 2 + 2) - 1

    index_GHM_row_1 = 0
    index_GHM_row_2 = nh * (the_degrees[0] + 1) - 1

    index_G_col_1 = 0
    index_G_col_2 = ns * (the_degrees[0] + 1) - 1

    index_u_1 = 0
    index_u_2 = nu * (the_degrees[0] + 1) - 1

    index12 = np.zeros([number_interval, 2])
    index_GHM_row12 = np.zeros([number_interval, 2])
    index_G_col12 = np.zeros([number_interval, 2])
    index_u12 = np.zeros([number_interval, 2])

    for iInterval in range(number_interval):
        index12[iInterval, 0] = index1
        index12[iInterval, 1] = index2
        K_size += the_degrees[iInterval] * 2 + 2

        index_GHM_row12[iInterval, 0] = index_GHM_row_1
        index_GHM_row12[iInterval, 1] = index_GHM_row_2
        GHM_row_size += the_degrees[iInterval] + 1

        index_G_col12[iInterval, 0] = index_G_col_1
        index_G_col12[iInterval, 1] = index_G_col_2
        G_col_size += the_degrees[iInterval] + 1

        index_u12[iInterval, 0] = index_u_1
        index_u12[iInterval, 1] = index_u_2

        if iInterval != number_interval - 1:
            index1 = index2 + 1
            index2 = index1 + ns * (the_degrees[iInterval + 1] * 2 + 2) - 1

            index_GHM_row_1 = index_GHM_row_2 + 1
            index_GHM_row_2 = index_GHM_row_1 + nh * (the_degrees[iInterval + 1] + 1) - 1

            index_G_col_1 = index_G_col_2 + 1
            index_G_col_2 = index_G_col_1 + ns * (the_degrees[iInterval + 1] + 1) - 1

            index_u_1 = index_u_2 + 1
            index_u_2 = index_u_1 + nu * (the_degrees[iInterval + 1] + 1) - 1

    K_size = K_size * ns
    GHM_row_size = GHM_row_size * nh
    G_col_size = G_col_size * ns

    index12 = index12.astype( int )
    index_GHM_row12 = index_GHM_row12.astype( int )
    index_G_col12 = index_G_col12.astype( int )
    index_u12 = index_u12.astype( int )

    K = np.zeros([K_size, K_size])
    G = np.zeros([GHM_row_size, G_col_size])
    H = np.zeros([GHM_row_size, G_col_size])
    M = np.zeros([GHM_row_size, GHM_row_size])
    kesi = np.zeros([K_size, GHM_row_size])
    zeta = np.zeros([K_size, 1])
    v = np.zeros([GHM_row_size, 1])

    for iInterval in range(number_interval):
        xi = x[index_G_col12[iInterval, 0] : index_G_col12[iInterval, 1] + 1]
        ui = u[index_u12[iInterval, 0] : index_u12[iInterval, 1] + 1]

        (tK, tkesi, tzeta, tG, tH, tM, tv) = CalSubMatrix(xi, ui, iInterval)

        K[index12[iInterval, 0]:index12[iInterval, 1]+1, index12[iInterval, 0]:index12[iInterval, 1]+1] += tK
        kesi[index12[iInterval, 0]:index12[iInterval, 1]+1, index_GHM_row12[iInterval, 0]:index_GHM_row12[iInterval, 1]+1] += tkesi
        zeta[index12[iInterval, 0]:index12[iInterval, 1]+1] += tzeta
        v[index_GHM_row12[iInterval, 0]:index_GHM_row12[iInterval, 1]+1] += tv
        G[index_GHM_row12[iInterval, 0]:index_GHM_row12[iInterval, 1]+1, index_G_col12[iInterval, 0]:index_G_col12[iInterval, 1]+1] += tG
        H[index_GHM_row12[iInterval, 0]:index_GHM_row12[iInterval, 1]+1, index_G_col12[iInterval, 0]:index_G_col12[iInterval, 1]+1] += tH
        M[index_GHM_row12[iInterval, 0]:index_GHM_row12[iInterval, 1]+1, index_GHM_row12[iInterval, 0]:index_GHM_row12[iInterval, 1]+1] += tM

    psi = -kesi
    phi = -zeta

    phi[index12[0, 0]:index12[0, 0]+ns] += lam[index_G_col12[0, 0]:index_G_col12[0, 0]+ns].reshape(-1, 1)
    phi[index12[-1, 1]-ns+1:index12[-1, 1]+1] += x[index_G_col12[-1, 1]-ns+1:index_G_col12[-1, 1]+1].reshape(-1, 1)

    if number_interval > 0:
        for iInterval in range(number_interval-1):
            K[index12[iInterval, 1]-ns+1:index12[iInterval, 1]+1, index12[iInterval, 1]+1:index12[iInterval, 1]+ns+1] -= np.identity(ns)
            K[index12[iInterval, 1]+1:index12[iInterval, 1]+ns+1, index12[iInterval, 1]-ns+1:index12[iInterval, 1]+1] -= np.identity(ns)
    
    K[0:ns, :] = 0.0
    K[0:ns, 0:ns] = np.identity(ns)
    K[-1-ns*(the_degrees[-1]+2)+1:-1-ns*(the_degrees[-1]+1)+1, :] = 0.0
    K[-1-ns*(the_degrees[-1]+2)+1:-1-ns*(the_degrees[-1]+1)+1, -1-ns*(the_degrees[-1]+2)+1:-1-ns*(the_degrees[-1]+1)+1] = np.identity(ns)

    psi[0:ns, :] = 0.0
    psi[-1-ns*(the_degrees[-1]+2)+1:-1-ns*(the_degrees[-1]+1)+1, :] = 0.0

    phi[0:ns] = x0.reshape(-1, 1)
    phi[-1-ns*(the_degrees[-1]+2)+1:-1-ns*(the_degrees[-1]+1)+1] = xf.reshape(-1, 1)

    invK = np.linalg.inv(K)

    invKpsi = np.dot(invK, psi)
    invKphi = np.dot(invK, phi)

    for iInterval in range(number_interval):
        if iInterval == 0:
            omega_x = np.copy( invKpsi[index12[iInterval, 0]:index12[iInterval, 0]+ns*(the_degrees[iInterval]+1)-1+1, :] )
            phi_x = np.copy( invKphi[index12[iInterval, 0]:index12[iInterval, 0]+ns*(the_degrees[iInterval]+1)-1+1] )
            omega_lam = np.copy( invKpsi[index12[iInterval, 0]+ns*(the_degrees[iInterval]+1):index12[iInterval, 1]+1, :] )
            phi_lam = np.copy( invKphi[index12[iInterval, 0]+ns*(the_degrees[iInterval]+1):index12[iInterval, 1]+1] )
        else:
            omega_x = np.vstack((omega_x, invKpsi[index12[iInterval, 0]:index12[iInterval, 0]+ns*(the_degrees[iInterval]+1)-1+1, :]))
            phi_x = np.vstack((phi_x, invKphi[index12[iInterval, 0]:index12[iInterval, 0]+ns*(the_degrees[iInterval]+1)-1+1]))
            omega_lam = np.vstack((omega_lam, invKpsi[index12[iInterval, 0]+ns*(the_degrees[iInterval]+1):index12[iInterval, 1]+1, :]))
            phi_lam = np.vstack((phi_lam, invKphi[index12[iInterval, 0]+ns*(the_degrees[iInterval]+1):index12[iInterval, 1]+1]))
    
    L = M - np.dot(G, omega_x) + np.dot(H, omega_lam)
    GAMA = -np.dot(G, phi_x) + np.dot(H, phi_lam) - v

    return (K, psi, phi, L, GAMA, invK)

def SolveMain(x0, xf, t0, tf):
    global ns, nu, nh, number_interval, the_degrees, index12
    global the_ends, the_etas, time_length, epsilonD, iter_max, cpt_number

    cpt_number = 0
    # the begin and end point of each interval
    cpt_index12 = np.empty([number_interval, 2])
    cpt1 = 0
    cpt2 = ns * ( the_degrees[0] + 1 ) - 1
    for iInterval in range(number_interval):
        cpt_number = cpt_number + the_degrees[iInterval] + 1
        cpt_index12[iInterval, 0] = cpt1
        cpt_index12[iInterval, 1] = cpt2
        if iInterval != number_interval - 1:
            cpt1 = cpt2 + 1
            cpt2 = cpt1 - 1 + (the_degrees[iInterval + 1] + 1) * ns
    
    cpt_index12 = cpt_index12.astype(int)

    (x_initial, u_initial, lam_initial) = InitialSolution(x0, xf, t0, tf)
    x_trial = np.copy(x_initial)
    u_trial = np.copy(u_initial)
    lam_trial = np.copy(lam_initial)

    iter = 0

    while 1:
        (K, psi, phi, L, GAMA, invK) = CalGlobalMatrix(x_trial, u_trial, lam_trial)

        sol = lcp.lemkelcp(L, GAMA)

        beta = sol[0].reshape(-1, 1)

        solu_1 = np.dot(invK, psi)
        solu = np.dot(solu_1, beta) + np.dot(invK, phi)

        x_new = np.zeros(x_trial.shape).reshape(-1, 1)
        lam_new = np.zeros(lam_trial.shape).reshape(-1, 1)

        for iInterval in range(number_interval):
            x_new[cpt_index12[iInterval, 0]:cpt_index12[iInterval, 1]+1] = solu[index12[iInterval, 0]:index12[iInterval, 0]-1+1+ns*(the_degrees[iInterval]+1)]
            lam_new[cpt_index12[iInterval, 0]:cpt_index12[iInterval, 1]+1] = solu[index12[iInterval, 0]+ns*(the_degrees[iInterval]+1):index12[iInterval, 1]+1]
        
        for iInterval in range(number_interval):
            if iInterval>0:
                lam_new[cpt_index12[iInterval, 0]:cpt_index12[iInterval, 0]+ns-1+1] = lam_new[cpt_index12[iInterval-1, 1]-ns+1:cpt_index12[iInterval-1, 1]+1]
            if iInterval<number_interval-1:
                x_new[cpt_index12[iInterval, 1]-ns+1:cpt_index12[iInterval, 1]+1] = x_new[cpt_index12[iInterval+1, 0]:cpt_index12[iInterval+1, 0]+ns-1+1]
        
        u_new = np.array([])
        for iCPT in range(cpt_number):
            xi = x_new[iCPT*ns:(iCPT+1)*ns]
            lami = lam_new[iCPT*ns:(iCPT+1)*ns]
            ui = u_trial[iCPT*nu:(iCPT+1)*nu]
            betai = beta[iCPT*nh:(iCPT+1)*nh]
            (tA, tB, tC, tD, tW, tV, tP, tR) = CoefficientMatrix(xi, ui)
            temp_1 = np.linalg.inv(tR)
            temp_2 = np.dot(tB.T, lami) + np.dot(tD.T, betai)
            ui = -np.dot(temp_1, temp_2)
            u_new = np.append(u_new, ui)
        
        delta_iter = np.linalg.norm(x_new - x_trial, ord=2) / np.linalg.norm(x_new, ord=2)
        print('delta is: {}'.format(delta_iter))

        x_trial = np.copy(x_new)
        u_trial = np.copy(u_new)
        lam_trial = np.copy(lam_new)
        
        if (delta_iter < epsilonD) or (iter == iter_max):
            break
        
        iter += 1
    return (x_trial, lam_trial, u_trial, cpt_number)

def LegendrePoly(n):
    if n == 0:
        pk = 1
    elif n == 1:
        pk = np.array([1, 0]).reshape(-1, 1)
    else:
        pkm2 = np.zeros((n+1, 1))
        pkm2[n, 0] = 1
        pkm1 = np.zeros((n+1, 1))
        pkm1[n-1, 0] = 1

        for k in range(1, n):

            pk = np.zeros((n+1, 1))
            for e in range (n-k-1, n, 2):
                pk[e, 0] = (2 * (k+1) - 1) * pkm1[e+1, 0] + (1-k-1) * pkm2[e, 0]
            
            pk[n, 0] = pk[n, 0] + (1-k-1) * pkm2[n, 0]
            pk = pk / (k+1)

            if (k+1) < n:
                pkm2 = np.copy( pkm1 )
                pkm1 = np.copy( pk )
        
        pk = np.flip( pk, 0 )

    if n%2 == 0:
        flag = 1
    else:
        flag = 0
    
    c = np.zeros((n+1, 1))
    for iN in range(n+1):
        c[iN, 0] = pk[iN, 0] * iN
        
    return (c, flag)

def JerkMin(x0, xf, t0, tf, step):
    global time_length, the_ends, the_etas

    time_length = tf - t0
    # the end point of each interval
    the_ends = np.arange(t0, tf, time_length/number_interval)
    the_ends = np.append(the_ends, tf)

    for i in range(number_interval):
        # the time length of each interval
        the_etas[i] = the_ends[i+1] - the_ends[i]

    (x, lam, u, cpt_number) = SolveMain(x0, xf, t0, tf)

    cpt1 = 0
    cpt2 = the_degrees[0]
    cpt_index12 = np.empty((number_interval, 2))
    for iInterval in range(number_interval):
        cpt_index12[iInterval, 0] = cpt1
        cpt_index12[iInterval, 1] = cpt2
        if iInterval != number_interval - 1:
            cpt1 = cpt2 + 1
            cpt2 = cpt1 + the_degrees[iInterval + 1]

    cpt_index12 = cpt_index12.astype(int)

    cpt = np.zeros((cpt_number, 1))
    for iInterval in range(number_interval):
        len = the_etas[iInterval]
        degree = the_degrees[iInterval]
        t1 = the_ends[iInterval]
        t2 = the_ends[iInterval + 1]
        (wg, _, xg) = LGL( degree )
        xg_new = (t1 + t2) / 2 + xg * len / 2
        cpt[cpt_index12[iInterval, 0]:cpt_index12[iInterval, 1]+1] = np.copy(xg_new).reshape(-1, 1)
    
    x_res = np.array([])
    for iCPT in range(cpt_number):
        x_res = np.append(x_res, x[iCPT * ns])

    # %%
    # refine the trajectory
    total_point = int(np.round(time_length / step)) + 1
    pointInterval = np.zeros(number_interval) + ( total_point - 1 ) / number_interval
    pointInterval = pointInterval.astype(int)

    x_ = np.array([])

    for iInterval in range( number_interval ):
        length_u = 2 / pointInterval[ iInterval ]
        time_tao = np.linspace(-1, 1, pointInterval[iInterval], endpoint=False)
    
        sub_x = np.zeros( (the_degrees[iInterval]+1, 1) )
        sub_x = x_res[cpt_index12[iInterval, 0] : cpt_index12[iInterval, 1] + 1]

        (_, rau, lg) = LGL( the_degrees[iInterval] )
        # c is the coefficient vector of Legendre polynomial
        (c, flag) = LegendrePoly( the_degrees[iInterval] )

        for i_tao in range( pointInterval[iInterval] ):
            L_e = 0
            for iN in range(1+flag, the_degrees[iInterval]+1+1):
                L_e = L_e + time_tao[i_tao] ** (iN - flag - 1) * c[iN - 1]
            
            mid_x = 0

            for l_degree in range(the_degrees[iInterval]+1):
                L_l = rau[l_degree]

                if abs(time_tao[i_tao] - lg[l_degree]) < 1e-7:
                    mid_x = sub_x[l_degree]
                else:
                    mid_x = mid_x + sub_x[l_degree] * (time_tao[i_tao]**2 - 1) * L_e / (  ( time_tao[i_tao] - lg[l_degree] ) * the_degrees[iInterval] * (the_degrees[iInterval] + 1) * L_l )
            
            x_ = np.append(x_, mid_x)
        
    x_ = np.append(x_, x_res[-1])

    t_stamp = np.linspace(t0, tf, int(total_point), endpoint=True)
    
    return (t_stamp, x_)

def TrajectoryGeneration(i_x0, i_xf, i_t0, i_tf, step, i_number_interval, i_degree, i_constraint_max, i_constraint_min):
    global number_interval, the_degrees, time_length, the_ends, the_etas, iter_max, cpt_number, constraint_max, constraint_min, epsilonD
    global x0, xf
    number_interval = i_number_interval
    the_degrees = np.ones(number_interval, dtype=int) * i_degree

    time_length = 0

    the_ends = np.empty(number_interval + 1)
    the_etas = np.empty(number_interval)

    epsilonD = 1e-6
    iter_max = 3

    cpt_number = 0

    constraint_max = i_constraint_max
    constraint_min = i_constraint_min

    x0 = i_x0
    xf = i_xf

    t0 = i_t0
    tf = i_tf

    (t_stamp, x) = JerkMin(x0, xf, t0, tf, step)

    x = x.reshape(1, -1)
    return (t_stamp, x)

def PathPlanning(x_list, v_list, a_list, t_list, max_list, min_list, step):
    nr_dof = x_list.shape[0]
    nr_point = x_list.shape[1]
    
    idx_1 = 0
    idx_2 = 1

    t_stamp = np.linspace(t_list[0], t_list[-1], int(t_list[-1]/step)+1, endpoint=True)
    
    position = np.zeros((nr_dof, len(t_stamp)))
    velocity = np.zeros((nr_dof, len(t_stamp)))
    acceleration = np.zeros((nr_dof, len(t_stamp)))
    jerk = np.zeros((nr_dof, len(t_stamp)))
    
    position[:, 0] = x_list[:, 0]
    velocity[:, 0] = v_list[:, 0]
    acceleration[:, 0] = a_list[:, 0]
    jerk[:, 0] = np.zeros(nr_dof)
            
    for i_point in range(1, nr_point ):
        
        idx_1 = idx_2
        idx_2 = int(t_list[i_point]/step) + 1
        
        for i_dof in range( nr_dof ):
            x_0 = np.array([x_list[i_dof, i_point-1], v_list[i_dof, i_point-1], a_list[i_dof, i_point-1]])
            x_f = np.array([x_list[i_dof, i_point], v_list[i_dof, i_point], a_list[i_dof, i_point]])
            
            [_, p_temp] = TrajectoryGeneration(x_0, x_f, t_list[i_point-1], t_list[i_point], step, 
                                               i_number_interval=2, i_degree=10, 
                                               i_constraint_max=max_list[i_dof, i_point-1], 
                                               i_constraint_min=min_list[i_dof, i_point-1])
            
            position[i_dof, idx_1:idx_2] = p_temp[0, 1:]
    for i in range(1, len(t_stamp)):
        velocity[:, i] = (position[:, i] - position[:, i-1]) / step
        acceleration[:, i] = (velocity[:, i] - velocity[:, i-1]) / step
        jerk[:, i] = (acceleration[:, i] - acceleration[:, i-1]) / step
    
    return (position, velocity, acceleration, jerk, t_stamp)        
            
if __name__ == '__main__':
    step = 0.01
    
    max_list = np.array([[50.0, 50.0, 50.0],
                         [50.0, 50.0, 50.0],
                         [50.0, 50.0, 50.0]]) * math.pi / 180
    min_list = np.array([[-50.0, -50.0, -50.0],
                         [-50.0, -50.0, -50.0],
                         [-50.0, -50.0, -50.0]]) * math.pi / 180
    
    t_list = [0.0, 1.0, 2.0, 2.2]
    t_list = np.array(t_list)
    # corresponding positions
    x_list = [[0.0, math.pi/4, 0.0, 0.0],
              [0.0, math.pi/5, 0.0, 0.0],
              [0.0, math.pi/6, 0.0, 0.0]]
    x_list = np.array(x_list)
    # corresponding velocities
    v_list = [[0.0, 5.0, 0.0, 0.0],
              [0.0, 5.0, 0.0, 0.0],
              [0.0, 5.0, 0.0, 0.0]]
    v_list = np.array(v_list)
    # None for free, concrete value for fixed
    a_list = [ [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0] ]
    a_list = np.array(a_list)
    
    [position, velocity, acceleration, jerk, t_stamp] = PathPlanning(x_list, v_list, a_list, t_list, max_list, min_list, step)
    
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