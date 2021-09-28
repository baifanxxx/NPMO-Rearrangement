#-*- coding:utf-8 -*-
import rospy, sys
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
import actionlib
import control_msgs.msg
import time

moveit_commander.roscpp_initialize(sys.argv)

# Create a goal to send (to the action server)
goal = control_msgs.msg.GripperCommandGoal()

arm = moveit_commander.MoveGroupCommander('maniplator')
robot_commander = moveit_commander.RobotCommander()

end_effector_link = arm.get_end_effector_link()

reference_frame = 'base_link'
arm.set_pose_reference_frame(reference_frame)

arm.allow_replanning(True)

arm.set_goal_position_tolerance(0.001)
arm.set_goal_orientation_tolerance(0.01)

arm.set_max_acceleration_scaling_factor(0.4)
arm.set_max_velocity_scaling_factor(0.4)
# self.pose[]


def move_to_joint(j0,j1,j2,j3,j4,j5):

	joint_goal = arm.get_current_joint_values()
	joint_goal[0] = j0
	joint_goal[1] = j1
	joint_goal[2] = j2
	joint_goal[3] = j3
	joint_goal[4] = j4
	joint_goal[5] = j5

	# The go command can be called with joint values, poses, or without any
	# parameters if you have already set the pose or joint target for the group
	arm.go(joint_goal, wait=True)
	rospy.sleep(1)


def move_to_position(x, y, z, qx, qy, qz, qw):
	target_pose = PoseStamped()
	target_pose.header.frame_id = reference_frame
	target_pose.header.stamp = rospy.Time.now()     
	target_pose.pose.position.x = x
	target_pose.pose.position.y = y
	target_pose.pose.position.z = z
	target_pose.pose.orientation.x = qx
	target_pose.pose.orientation.y = qy
	target_pose.pose.orientation.z = qz
	target_pose.pose.orientation.w = qw

	arm.set_start_state_to_current_state()

	arm.set_pose_target(target_pose, end_effector_link)

	traj = arm.plan()
	arm.execute(traj)
	rospy.sleep(1)

	arm.go()
	rospy.sleep(1)

def cartesian_move_to_position(x, y, z, qx, qy, qz, qw):
	target_pose = PoseStamped()
	target_pose.header.frame_id = reference_frame
	target_pose.header.stamp = rospy.Time.now()
	target_pose.pose.position.x = x
	target_pose.pose.position.y = y
	target_pose.pose.position.z = z
	target_pose.pose.orientation.x = qx
	target_pose.pose.orientation.y = qy
	target_pose.pose.orientation.z = qz
	target_pose.pose.orientation.w = qw

	start_pose = arm.get_current_pose(end_effector_link).pose
	waypoints = []
	waypoints.append(start_pose)
	waypoints.append(target_pose.pose)

	fraction = 0.0  
	maxtries = 1000 
	attempts = 0  

	arm.set_start_state_to_current_state()

	while fraction < 1.0 and attempts < maxtries:

		(plan, fraction) = arm.compute_cartesian_path(
			waypoints,  
			0.01,  
			0.0,  
			True)  

		attempts += 1

		if attempts % 10 == 0:
			rospy.loginfo("Still trying after " + str(attempts) + " attempts...")

	if fraction == 1.0:
		rospy.loginfo("Path computed successfully. Moving the arm.")
		arm.execute(plan)
		rospy.sleep(1)

		arm.go()
		rospy.sleep(1)
		rospy.loginfo("Path execution complete.")

	else:
		rospy.loginfo(
			"Path planning failed with only " + str(fraction) + " success after " + str(maxtries) + " attempts.")

	rospy.sleep(1)



def move_home():
        arm.set_named_target('home')
        arm.go()

def get_end_effector_link():
	return arm.get_end_effector_link()

def get_current_state():
	return robot_commander.get_current_state()
