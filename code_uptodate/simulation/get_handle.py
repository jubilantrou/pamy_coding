import pam_mujoco
from scipy.spatial.transform import Rotation

def get_handle(mujoco_id='Pamy_sim', mode='pressure', generation='second', rotation=90, position=(0.2, 0.0, 1.21)):
    # It is assumed that the waiting instance of pam_mujoco was started already using this same mujoco_id.

    # the o80 segment id of the robot
    robot_segment_id = "robot"

    # the robot control mode
    if mode == 'pressure':
        control = pam_mujoco.MujocoRobot.PRESSURE_CONTROL
    elif mode == 'joint':
        control = pam_mujoco.MujocoRobot.JOINT_CONTROL
    
    # the generation of the robot
    if generation == 'first':
        robot_type = pam_mujoco.RobotType.PAMY1
    elif generation == 'second':
        robot_type = pam_mujoco.RobotType.PAMY2

    # creating the robot
    robot = pam_mujoco.MujocoRobot(
        robot_type = robot_type,
        segment_id = robot_segment_id,
        control = control,
        orientation = Rotation.from_euler('z',rotation,degrees=True),
        position = position
        ) 
    # constructing the handle
    table = pam_mujoco.MujocoTable("table", position=(0.4, 1.57, 0.755))
    handle = pam_mujoco.MujocoHandle(
        mujoco_id, robot1=robot, table=table, graphics=True, accelerated_time=False
    )

    return handle