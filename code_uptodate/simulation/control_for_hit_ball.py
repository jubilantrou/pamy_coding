import threading

period_backend = 1.0 / frequency_backend  # period of backend
period_frontend = 1.0 / frequency_frontend  # period of frontend
t = 1 / frequency_frontend
iterations_per_command = int( period_frontend / period_backend )  # sychronize the frontend and the backend

pid = np.array([[-13000, 0, -300],
                [80000, 0, 300],
                [-5000, -8000, -100],
                [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])

    def HitBall(self, target, T_go, T_back=1.0, frequency_frontend=100, frequency_backend=500):
        # target and T_go will be updated on real time
        # usually we use T_back = 1.2s to return back
        def sending_command(pressure_ago,pressure_ant, iteration):
            self.frontend.add_command(pressure_ago, pressure_ant,
                                        o80.Iteration(iteration + iterations_per_command - 1),
                                        o80.Mode.QUEUE)

            self.frontend.pulse_and_wait()
            iteration += iterations_per_command
            return iteration
        
        def controlling(time_point, real_trajectory, angle_delta_pre, res_i ):
            time_point += 1  # begin from 0
            # read the actual current states
            theta = np.array( self.frontend.latest().get_positions() )
            theta_dot = np.array( self.frontend.latest().get_velocities() )
            pressures = np.array( self.frontend.latest().get_observed_pressures() )
            # the actual ago pressures and ant pressures
            pressure_ago = pressures[:, 0]
            pressure_ant = pressures[:, 1]

            real_trajectory = np.append(real_trajectory, theta)

            # read T_back and target from shared memory
            trajectory = RobotGeometry.RacketTrajectory(time_point=time_point, angle=theta, angle_dot=theta_dot, target=target)
            ff = NNGeneralization( trajectory )

            angle_delta = (trajectory[:, 1] + angle_initial - theta).reshape(len(self.dof_list), -1)
            res_d = ( angle_delta - angle_delta_pre ) / t
            res_i += angle_delta * t

            feedback = pid[:, 0].reshape(len(self.dof_list), -1) * angle_delta\
                     + pid[:, 1].reshape(len(self.dof_list), -1) * res_i\
                     + pid[:, 2].reshape(len(self.dof_list), -1) * res_d
            
            angle_delta_pre = np.copy( angle_delta )

            # if i == 0:
            #     fb = np.copy( feedback )
            # else:
            #     fb = np.hstack((fb, feedback))
            
            diff = ff[:, i] + fb[:, i]

            pressure_ago = int( self.anchor_ago_list + diff )
            pressure_ant = int( self.anchor_ant_list - diff )
            
            # do not control the last dof
            pressure_ago[3] = self.anchor_ago_list[3]
            pressure_ant[3] = self.anchor_ant_list[3]

            if time_point == (T_go + T_back) * 100:
                fail_flag = True
            
            return time_point, real_trajectory, angle_delta_pre, res_i, fail_flag
        
        error_x = np.array([])
        error_y = np.array([])
        fb = np.array([])

        angle_delta_pre = np.zeros( len(self.dof_list) ).reshape(len(self.dof_list), -1)
        res_i = 0

        fail_flag = False  # if fail to hit the ball
        time_point = -1

         # save the real trajectory
        real_trajectory = np.array([])

        theta = np.array( self.frontend.latest().get_positions() )
        theta_dot = np.array( self.frontend.latest().get_velocities() )
        pressures = np.array( self.frontend.latest().get_observed_pressures() )
        # the actual ago pressures and ant pressures
        pressure_ago = pressures[:, 0]
        pressure_ant = pressures[:, 1]

        # read the current iteration
        iteration_reference = self.frontend.latest().get_iteration()  
        # set the beginning iteration number
        iteration_begin = iteration_reference + 1500

        iteration = iteration_begin

        while not fail_flag:
            if target is not None:
                
                thread_0 = threading.Thread(targer=sending_command, args=(pressure_ago, pressure_ant, iteration))
                thread_1 = threading.Thread(targer=controlling,args=(time_point, real_trajectory, angle_delta_pre, res_i ))

                thread_0.start()
                thread_1.start()

                self.frontend.add_command(pressure_ago, pressure_ant,
                                          o80.Iteration(iteration),
                                          o80.Mode.QUEUE)
        
                self.frontend.pulse()

                