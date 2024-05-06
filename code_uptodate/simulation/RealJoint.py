'''
This script is used to save the inverse model information for each DoF, 
which support computing desired feedforward inputs from the reference trajectory 
but are not used in the procedure yet.
'''
import numpy as np

class Joint:

    def __init__(self, frontend, dof, anchor_ago, anchor_ant, 
                 num, den, order_num, order_den, ndelay, 
                 pid, strategy):
                                
        self.frontend = frontend
        self.dof = dof
        self.anchor_ago = anchor_ago
        self.anchor_ant = anchor_ant
        
        self.order_num = order_num
        self.order_den = order_den
        self.num = num[0:order_num+1]
        self.den = den[0:order_den+1]
        self.delay = ndelay
        
        self.pid = pid
        self.strategy = strategy

    def Feedforward(self, y):
        '''
        here y should be the relative angles
        '''
        u_ago = []
        u_ant = []
        ff = []

        for i in range(0, len( y )):
            sum_num = 0
            for Nr in range(self.order_num + 1):
                a = i + self.delay - Nr
                if a >= len(y):
                    a = len(y) - 1
                if a >= 0:
                    term = self.num[Nr] * (y[a] - y[i])
                else:
                    term = self.num[Nr] * 0.0
                sum_num += term
            
            sum_den = 0
            for Nr in range(1, self.order_den + 1):
                a = i - Nr
                if a >= 0:
                    term = self.den[Nr] * ff[a]
                else:
                    term = self.den[Nr] * 0.0
                sum_den += term

            feedforward = sum_num - sum_den
            pressure_steady_ago = self.anchor_ago
            pressure_steady_ant = self.anchor_ant

            ff.append(feedforward)
            u_ago.append(pressure_steady_ago)
            u_ant.append(pressure_steady_ant)

        u_ago = np.array(u_ago)
        u_ant = np.array(u_ant)
        ff = np.array(ff)

        return(u_ago, u_ant, ff)
