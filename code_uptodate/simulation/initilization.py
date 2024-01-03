'''
This script is used to initilize the posture.
'''
import math
import numpy as np
import o80
import os
from get_handle import get_handle
# %% connect to the simulation and initilize the posture
handle           = get_handle(mode='joint')
frontend         = handle.frontends["robot"]
joints           = (0.0, math.pi / 6.0, math.pi / 6.0, 0.0)
joint_velocities = (0, 0, 0, 0)
duration         = o80.Duration_us.seconds(2)
frontend.add_command(joints, joint_velocities, duration, o80.Mode.QUEUE)
frontend.pulse()
print('...completed')