'''
This script is used to warm the robot arm up.
'''
import PAMY_CONFIG
import math
import os
import numpy as np
import o80
import o80_pam
import matplotlib.pyplot as plt
# %%
frontend                   = o80_pam.FrontEnd("real_robot")
Geometry                   = PAMY_CONFIG.build_geometry()
Pamy                       = PAMY_CONFIG.build_pamy(frontend=frontend)
# %%
# Pamy.AngleInitialization(Geometry.initial_posture)
# Pamy.AngleInitialization(np.array([30*math.pi/180, 60*math.pi/180, 45*math.pi/180, 0.0 ]))
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())