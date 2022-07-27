#!/usr/bin/env python
# Python 2/3 compatibility imports
from __future__ import print_function

import json
import math
import sys
import numpy as np
import rospy
from geometry_msgs.msg import Quaternion
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from tf import transformations
import roslib

from moveit_msgs.msg import Constraints, OrientationConstraint

roslib.load_manifest('nao_description')

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi


    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

try:
    import motion
    from naoqi import ALProxy
except ImportError:
    print("Error importing NaoQI. Please make sure that Aldebaran's NaoQI API is in your PYTHONPATH.")


def get_R():
    # axes rotation from world_pose origin to base_link
    # rotation is counterclockwise if θ is positive (e.g. 90°)
    a = math.pi / 2  # -90° in x,
    b = math.pi / 2  # -90° in z

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(a), math.sin(a)],
        [0.0, -math.sin(a), math.cos(a)]
    ])
    Ry = np.array([
        [math.cos(b), 0.0, math.sin(b)],
        [0.0, 1.0, 0.0],
        [-math.sin(b), 0.0, math.cos(b)]
    ])

    return np.matmul(Rx, Ry)  # z is eye -> no rotation needed


def rmtx_to_homo(mtx3x3):
    mh = np.hstack((mtx3x3, np.array([[0], [0], [0]])))
    mh = np.vstack((mh, np.array([0, 0, 0, 1])))
    return mh


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


def vec_length(v: np.array):
    return np.sqrt(sum(i ** 2 for i in v))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def look_at(eye: np.array, target: np.array):
    axis_z = normalize((eye - target))

    if vec_length(axis_z) == 0:
        axis_z = np.array((0, -1, 0))

    axis_x = np.cross(np.array((0, 0, 1)), axis_z)

    if vec_length(axis_x) == 0:
        axis_x = np.array((1, 0, 0))

    axis_y = np.cross(axis_z, axis_x)
    rot_matrix = np.matrix([axis_x, axis_y, axis_z]).transpose()
    return rot_matrix


class PocInterface(object):
    """PocInterface"""

    def __init__(self):
        super(PocInterface, self).__init__()

        # initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("pynao", anonymous=True)
        # provides information such as the robot's kinematic model and the robot's current joint states
        self.robot = moveit_commander.RobotCommander()
        # provides a remote interface for getting, setting, and updating the robot's internal understanding of the
        # surrounding world:
        self.scene = moveit_commander.PlanningSceneInterface()
        # Create a `DisplayTrajectory`_ ROS publisher which is used to display trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        self.display_trajectory_publisher = display_trajectory_publisher

        # END_SUB_TUTORIAL

        # BEGIN_SUB_TUTORIAL basic_info
        ##
        # Getting Basic Information
        # ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get a list of all the groups in the robot:
        self.group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", self.robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")
        # END_SUB_TUTORIAL

        # Misc variables
        self.box_name = "box"

    def go_to_joint_state(self):
        move_group = moveit_commander.MoveGroupCommander("left_arm")
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 8
        joint_goal[2] = 0
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 6  # 1/6 of a turn
        joint_goal[6] = 0

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        # END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self):
        # Planning to a Pose Goal
        # ^^^^^^^^^^^^^^^^^^^^^^^
        move_group = moveit_commander.MoveGroupCommander("left_arm")
        current_pose = move_group.get_current_pose()
        print('================== Current pose:\n', current_pose)

        # We can plan a motion for this group to a desired pose for the
        # end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        # pose_goal = current_pose.pose

        pose_goal.position.x = 0.1229850771152338
        pose_goal.position.y = 0.06995081842192753
        pose_goal.position.z = 0.0349783006680724
        # pose_goal.orientation.x = 0.06726783742876526
        # pose_goal.orientation.y = 0.4643996278369603
        # pose_goal.orientation.z = -0.3404691033131412
        # pose_goal.orientation.w = 0.8147937244491366
        pose_goal.orientation = current_pose.pose.orientation

        # move_group.set_pose_target(pose_goal)
        # move_group.set_random_target()
        # Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)

        print("============ Waiting while RVIZ displays plan...")
        rospy.sleep(5)

        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

        # For testing:
        current_pose = move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def send_body_pose(self, data):
        rps = "13"
        rpi = "15"
        lps = "14"
        lpi = "16"

        mg_left = moveit_commander.MoveGroupCommander("left_arm")
        mg_left.set_pose_reference_frame("base_link")
        mg_left.set_planner_id("RRTkConfigDefault")
        # mg_left.set_planning_time(5)
        # mg_left.set_goal_position_tolerance(0.01)
        # mg_left.set_goal_tolerance(0.01)
        mg_right = moveit_commander.MoveGroupCommander("right_arm")
        mg_right.set_pose_reference_frame("base_link")
        mg_right.set_planner_id("RRTkConfigDefault")
        # mg_right.set_planning_time(5)
        # mg_right.set_goal_position_tolerance(0.01)
        # mg_right.set_goal_tolerance(0.01)

        for i, row in enumerate(data):
            if i % 2 == 0:
                mg_left.clear_pose_targets()
                mg_right.clear_pose_targets()
                self.scene.remove_world_object()
                # convert input data
                # lfs = self.convert_point(row[lps])
                # lfi = self.convert_point(row[lpi])
                # rfs = self.convert_point(row[rps])
                # rfi = self.convert_point(row[rpi])
                lfs = row[lps]
                lfi = row[lpi]
                rfs = row[rps]
                rfi = row[rpi]
                # orientations
                eyel = np.array([lfs[0], lfs[1], lfs[2]])
                targetl = np.array([lfi[0], lfi[1], lfi[2]])
                eyer = np.array([rfs[0], rfs[1], rfs[2]])
                targetr = np.array([rfi[0], rfi[1], rfi[2]])

                lor = look_at(eyel, targetl)
                ror = look_at(eyer, targetr)

                # left
                lp_goal = geometry_msgs.msg.PoseStamped()
                lp_goal.header.seq = i
                lp_goal.header.frame_id = "base_link"
                lp_goal.pose.position.x, lp_goal.pose.position.y, lp_goal.pose.position.z = self.convert_point(row[lpi])
                # to homogenous coordinates orientation
                # lp_goal.pose.orientation = Quaternion(*transformations.quaternion_from_matrix(rmtx_to_homo(lor)))
                angl = transformations.euler_from_matrix(rmtx_to_homo(lor))
                print('angl:\n', angl)
                # angl = (angl[0] - 3.1415, angl[1], angl[2])
                angl = (-angl[0], angl[1], angl[2])
                lp_goal.pose.orientation = Quaternion(*transformations.quaternion_from_euler(*angl))
                print("L orientation: ", lp_goal.pose.orientation)
                mg_left.set_joint_value_target(lp_goal, arg3=True)
                # mg_left.set_pose_target(lp_goal)
                print('lp_goal:\n', lp_goal)

                # right
                rp_goal = geometry_msgs.msg.PoseStamped()
                rp_goal.header.seq = i
                rp_goal.header.frame_id = "base_link"
                rp_goal.pose.position.x, rp_goal.pose.position.y, rp_goal.pose.position.z = self.convert_point(row[rpi])
                print("equal points?? -> ", [self.convert_point(row[rpi])], self.rotate_point(row[rpi]))
                print("result: ?? -> ", all_close([self.convert_point(row[rpi])], self.rotate_point(row[rpi]), 0.000001))
                # to homogenous coordinates orientation
                rp_goal.pose.orientation = Quaternion(*transformations.quaternion_from_matrix(rmtx_to_homo(ror)))
                print("R orientation: ", rp_goal.pose.orientation)
                mg_right.set_joint_value_target(rp_goal, arg3=True)
                # mg_right.set_pose_target(rp_goal)
                print('rp_goal:\n', rp_goal)

                # compute the plan and execute it
                mg_left.go(wait=True)
                mg_right.go(wait=True)
                # print("Waiting while RVIZ displays plan...")
                # rospy.sleep(0.5)

                # stop
                mg_left.stop()
                mg_right.stop()
                self.add_box("boxR", rp_goal)
                self.add_box("boxL", lp_goal)
                rospy.sleep(0.5)
                # input("=> Press Enter for next keyframe...")
                # clear targets
                # print('result:', all_close(lp_goal, mg_left.get_current_pose(), 0.00001)
                #       and all_close(rp_goal.pose.position, mg_right.get_current_pose().pose.position, 0.00001))
        return 1

    def draw_boxes(self, data):
        for row in data:
            for key, itm in row.items():
                box_pose = geometry_msgs.msg.PoseStamped()
                box_pose.header.frame_id = "base_link"
                box_pose.pose.position.x, box_pose.pose.position.y, box_pose.pose.position.z = self.convert_point(itm)
                if key == "15":
                    print("box pose for 15:", box_pose.pose.position)
                # print('box_pose:\n', box_pose.pose.position)
                self.add_box("box" + key, box_pose)
            rospy.sleep(0.2)
            # input("=> Press Enter for next keyframe...")

    def box_and_pose(self, data):
        rgoal = geometry_msgs.msg.PoseStamped()
        lgoal = geometry_msgs.msg.PoseStamped()
        rgoal.header.frame_id = "base_link"
        lgoal.header.frame_id = "base_link"

        # right
        # point above the head
        rgoal.pose.position.x = 0.0199795786822619
        rgoal.pose.position.y = -0.12291418520949536
        rgoal.pose.position.z = 0.25834824560873787
        # orientation for that point:
        rgoal.pose.orientation.x = 0.31966536767769327
        rgoal.pose.orientation.y = -0.6391995880378866
        rgoal.pose.orientation.z = 0.2547530498973046
        rgoal.pose.orientation.w = 0.6514129434757209

        # left
        lgoal = geometry_msgs.msg.PoseStamped()
        lgoal.header.frame_id = "base_link"
        # point above the left head
        lgoal.pose.position.x = 0.0199795786822619
        lgoal.pose.position.y = 0.12291418520949536
        lgoal.pose.position.z = 0.25834824560873787
        # orientation for that point:
        lgoal.pose.orientation.x = -0.31966536767769327
        lgoal.pose.orientation.y = -0.6391995880378866
        lgoal.pose.orientation.z = -0.2547530498973046
        lgoal.pose.orientation.w = 0.6514129434757209

        mgr = moveit_commander.MoveGroupCommander("right_arm")
        mgl = moveit_commander.MoveGroupCommander("left_arm")
        mgr.set_pose_reference_frame("base_link")
        mgl.set_pose_reference_frame("base_link")
        print("LEFT arm pose -> ", mgl.get_current_pose())
        # print("left arm pose -> ", moveit_commander.MoveGroupCommander("left_arm").get_current_pose())
        # mg.set_planner_id("RRTConnectkConfigDefault")
        #  planners = (SBLkConfigDefault, ESTkConfigDefault, LBKPIECEkConfigDefault, BKPIECEkConfigDefault,
        # KPIECEkConfigDefault, RRTkConfigDefault, RRTConnectkConfigDefault, RRTstarkConfigDefault,
        # TRRTkConfigDefault, PRMkConfigDefault, PRMstarkConfigDefault)
        # mg.set_goal_position_tolerance(0.00001)
        print("rf:", mgr.get_pose_reference_frame())

        # mg.set_joint_value_target(goal, arg3=True)
        mgr.set_pose_target(rgoal)
        mgl.set_pose_target(lgoal)
        # mg.set_position_target([0.0199795786822619, -0.12291418520949536, 0.25834824560873787])
        # mgr.plan()
        # mgl.plan()
        mgr.go(wait=True)
        mgl.go(wait=True)
        # rospy.sleep(5)
        self.add_box("boxD", rgoal)

    def add_box(self, box_name, pose):
        scene = self.scene
        scene.add_box(box_name, pose, size=(0.015, 0.015, 0.015))

    def test_both_arms(self):
        mg_both = moveit_commander.MoveGroupCommander("both_arms")
        print("arms pose l -> ", mg_both.get_current_pose(end_effector_link="l_wrist"))
        print("arms pose r -> ", mg_both.get_current_pose(end_effector_link="r_wrist"))

        # right
        rgoal = geometry_msgs.msg.PoseStamped()
        rgoal.header.frame_id = "base_link"
        # point above the right head
        rgoal.pose.position.x = 0.0199795786822619
        rgoal.pose.position.y = -0.12291418520949536
        rgoal.pose.position.z = 0.25834824560873787
        # orientation for that point:
        rgoal.pose.orientation.x = 0.31966536767769327
        rgoal.pose.orientation.y = -0.6391995880378866
        rgoal.pose.orientation.z = 0.2547530498973046
        rgoal.pose.orientation.w = 0.6514129434757209

        # left
        lgoal = geometry_msgs.msg.PoseStamped()
        lgoal.header.frame_id = "base_link"
        # point above the left head
        lgoal.pose.position.x = 0.0199795786822619
        lgoal.pose.position.y = 0.12291418520949536
        lgoal.pose.position.z = 0.25834824560873787
        # orientation for that point:
        lgoal.pose.orientation.x = -0.31966536767769327
        lgoal.pose.orientation.y = -0.6391995880378866
        lgoal.pose.orientation.z = -0.2547530498973046
        lgoal.pose.orientation.w = 0.6514129434757209

        mg_both.set_joint_value_target(lgoal, arg3=True)
        # mg_both.set_pose_target(rgoal, end_effector_link="r_wrist")
        s1, plan1, _, _ = mg_both.plan()
        mg_both.set_joint_value_target(rgoal, arg3=True)
        # mg_both.set_pose_target(lgoal, end_effector_link="l_wrist")
        s11, plan2, _, _ = mg_both.plan()
        # mg_both.set_joint_value_target(lgoal, arg2="l_wrist", arg3=True)
        # mg_both.set_joint_value_target()
        # mg.set_position_target([0.0199795786822619, -0.12291418520949536, 0.25834824560873787])
        # mg_both.plan(target=rgoal)
        mg_both.go(wait=True)
        mg_both.stop()
        mg_both.clear_pose_targets()

        # rospy.sleep(5)
        self.add_box("boxR", rgoal)
        self.add_box("boxL", lgoal)

    @staticmethod
    def convert_point(coords):
        # r = 0.45
        r = 1.0
        # h = 0.1
        h = 0.0
        x = - float(coords[2]) * r  # x' = -rz
        y = - float(coords[0]) * r  # y' = -rx
        z = (- float(coords[1]) * r) - h  # z' = -y-h (to lower the center a bit)
        # return x, y, z
        return x, y, z

    @staticmethod
    def rotate_point(coords):
        return np.matmul(get_R(), coords)


def main():
    try:
        print("\n-----------------------------------------")
        print("Welcome to poc")
        print("-----------------------------------------\n")

        poc = PocInterface()

        # print("-- quaternion_from_euler(0, 0, 0): ", transformations.quaternion_from_euler(0, 0, 0))

        print("=== Reading file")
        with open("/home/mattia/ws_moveit/src/pynao/mp_out/bp2d/kpts_world_2D3.json", 'r') as fp:
            data = json.load(fp)

        # clean scene
        rospy.sleep(1)
        poc.scene.remove_world_object()
        res = input("Enter.. \n"
                    "1 to execute a movement using a pose goal,\n"
                    "2 to read and send points from file,\n"
                    "3 to view points as boxes,\n"
                    "4 to test a point with a single box and arm, \n"
                    "5 to test both arms group: ")

        if int(res) == 1:
            print("===> Executing a movement using a pose goal ...")
            poc.go_to_pose_goal()
        elif int(res) == 2:
            print("===> Sending poses ...")
            poc.send_body_pose(data)
        elif int(res) == 3:
            print("===> Boxing poses ...")
            poc.draw_boxes(data)
        elif int(res) == 4:
            print("===> Draw boxes ...")
            poc.box_and_pose(data)
        elif int(res) == 5:
            print("===> Both arms test ...")
            poc.test_both_arms()
        else:
            print("Invalid number inserted!")

        print("============ Complete! ============")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
