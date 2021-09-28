# Borrowed and modified from the kinova-ros examples.

#-*- coding:utf-8 -*-
import rospy, sys
import actionlib
import control_msgs.msg


def set_finger_positions(finger_positions):
    """Send a gripper goal to the action server."""
    global gripper_client

    goal = control_msgs.msg.GripperCommandGoal()
    goal.command.position = finger_positions  # From 0.0 to 0.8
    goal.command.max_effort = -1.0  # Do not limit the effort

    # Wait until the action server has been started and is listening for goals
    gripper_client.wait_for_server()

    gripper_client.send_goal(goal)
    if gripper_client.wait_for_result(rospy.Duration(5.0)):
        return gripper_client.get_result()
    else:
        gripper_client.cancel_all_goals()
        rospy.logwarn('        the gripper action timed-out')
        return None

action_address = '/gripper_controller/gripper_cmd'
gripper_client = actionlib.SimpleActionClient(action_address, control_msgs.msg.GripperCommandAction)
