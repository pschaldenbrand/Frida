#! /usr/bin/env python

import rospy
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
    SolvePositionFK,
    SolvePositionFKRequest,
)
import intera_interface
from intera_interface import CHECK_VERSION

class Robot:
    def __init__(self, debug, node_name="painting"):
        self.debug = debug

        rospy.init_node(node_name)

    def debug(self, msg):
        if self.debug:
            print(msg)

    def good_morning_robot(self):
        raise Exception("This method must be implemented")

    def good_night_robot(self):
        raise Exception("This method must be implemented")

    def move_to_joint_positions(self, position):
        raise Exception("This method must be implemented")


class Sawyer(Robot):
    def __init__(self, debug=True):
        super().__init__(debug)

    def good_morning_robot(self):
        self.debug("Getting robot state... ")
        rs = intera_interface.RobotEnable(False)
        init_state = rs.state().enabled
        self.debug("Enabling robot... ")
        rs.enable()

        def clean_shutdown():
            """
            Exits example cleanly by moving head to neutral position and
            maintaining start state
            """
            self.debug("\nExiting example...")
            limb = intera_interface.Limb(synchronous_pub=True)
            limb.move_to_neutral(speed=.15)

        rospy.on_shutdown(clean_shutdown)
        self.debug("Excecuting... ")

        return rs

    def good_night_robot(self):
        """ Tuck it in, read it a story """
        rospy.signal_shutdown("Example finished.")
        self.debug("Done")

    def inverse_kinematics(self, position, orientation, seed_position=None, debug=False):
        """
        args:
            position=(x,y,z)
            orientation=(x,y,z,w)
        kwargs:
            seed_position={'right_j0':float, 'right_j1':float, ...}
        return:
            dict{'right_j0',float} - dictionary of joint to joint angle
        """
        ns = "ExternalTools/right/PositionKinematicsNode/IKService"
        iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        ikreq = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        pose = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=position[0],
                    y=position[1],
                    z=position[2],
                ),
                orientation=Quaternion(
                    x=orientation[0],
                    y=orientation[1],
                    z=orientation[2],
                    w=orientation[3],
                ),
            ),
        )
        # Add desired pose for inverse kinematics
        ikreq.pose_stamp.append(pose)
        # Request inverse kinematics from base to "right_hand" link
        ikreq.tip_names.append('right_hand')

        if (seed_position is not None):
            # Optional Advanced IK parameters
            rospy.loginfo("Running Advanced IK Service Client example.")
            # The joint seed is where the IK position solver starts its optimization
            ikreq.seed_mode = ikreq.SEED_USER
            seed = JointState()
            seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                         'right_j4', 'right_j5', 'right_j6']
            seed.position = [seed_position['right_j0'], seed_position['right_j1'],
                             seed_position['right_j2'], seed_position['right_j3'],
                             seed_position['right_j4'], seed_position['right_j5'],
                             seed_position['right_j6']]
            ikreq.seed_angles.append(seed)

            # # Once the primary IK task is solved, the solver will then try to bias the
            # # the joint angles toward the goal joint configuration. The null space is
            # # the extra degrees of freedom the joints can move without affecting the
            # # primary IK task.
            # ikreq.use_nullspace_goal.append(True)
            # # The nullspace goal can either be the full set or subset of joint angles
            # goal = JointState()
            # goal.name = ['right_j1', 'right_j2', 'right_j3']
            # goal.position = [0.1, -0.3, 0.5]
            # ikreq.nullspace_goal.append(goal)
            # # The gain used to bias toward the nullspace goal. Must be [0.0, 1.0]
            # # If empty, the default gain of 0.4 will be used
            # ikreq.nullspace_gain.append(0.4)

        try:
            rospy.wait_for_service(ns, 5.0)
            resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False


        # if resp.result_type[0] == resp.IK_IN_COLLISION:
        #     print('COOLLISSSIIIIONNN')

        # Check if result valid, and type of seed ultimately used to get solution
        if (resp.result_type[0] > 0):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp.result_type[0], 'None')
            if debug:
                rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
                      (seed_str,))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(list(zip(resp.joints[0].name, resp.joints[0].position)))
            if debug:
                rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
                rospy.loginfo("------------------")
                rospy.loginfo("Response Message:\n%s", resp)
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            rospy.logerr("Result Error %d", resp.result_type[0])
            return False
        # Result to dictionary of joint angles
        pos = {}
        for i in range(len(resp.joints[0].name)):
            name = resp.joints[0].name[i]
            position = resp.joints[0].position[i]
            # print(name, position)
            pos[name] = position
        return pos

    def move_to_joint_positions(self, position, timeout=3, speed=0.1):
        """
        args:
            dict{'right_j0',float} - dictionary of joint to joint angle
        """
        # rate = rospy.Rate(100)
        try:
            limb = intera_interface.Limb(synchronous_pub=False)
            # limb.move_to_neutral()

            # print('Positions:', position)
            limb.set_joint_position_speed(speed=speed)
            limb.move_to_joint_positions(position, timeout=timeout,
                                         threshold=0.008726646*1)
            limb.set_joint_position_speed(speed=.1)
            # rate.sleep()
        except Exception as e:
            print('Exception while moving robot:\n', e)


    def display_image(file_path):
        head_display = intera_interface.HeadDisplay()
        # display_image params:
        # 1. file Path to image file to send. Multiple files are separated by a space, eg.: a.png b.png
        # 2. loop Display images in loop, add argument will display images in loop
        # 3. rate Image display frequency for multiple and looped images.
        head_display.display_image(file_path, False, 100)
    def display_frida(self):
        import rospkg
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        ros_dir = rospack.get_path('paint')
        self.display_image(os.path.join(str(ros_dir), 'scripts', 'frida.jpg'))