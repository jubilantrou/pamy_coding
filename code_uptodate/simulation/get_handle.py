import pam_mujoco
from scipy.spatial.transform import Rotation

def get_handle(mujoco_id='pamy_sim', mode='pressure', generation='second', rotation_z_robot=90, position=(0.0, 1.7, 0.755)):
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
        position = (0.0, 0.0, 1.21),
        orientation = Rotation.from_euler('z',rotation_z_robot,degrees=True),
        ) 
    # constructing the handle
    table = pam_mujoco.MujocoTable("table", position=position)
    hit_point = pam_mujoco.MujocoItem("hit_point", control=pam_mujoco.MujocoItem.CONSTANT_CONTROL, color=(0,0,1,1))
    handle = pam_mujoco.MujocoHandle(
        mujoco_id, robot1=robot, table=table, hit_points=(hit_point,), graphics=True, accelerated_time=False
    )

    return handle