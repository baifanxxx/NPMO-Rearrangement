# -*- coding:utf-8 -*-
#!/usr/bin/python2
import sys
import cv2
from utils.env import ENV
import numpy as np
from copy import deepcopy
import time
from models import Actor,Critic
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image as PImage
from helpers.transforms import get_trans, trans_pose
import rospy, sys
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
import control_msgs.msg
import copy


def route_length(route):
    res = 0
    lx = None
    ly = None
    for _ in route:
        x,y,_,_ = _
        if lx is not None:
            res += abs(lx-x) + abs(ly-y)
        lx = x
        ly = y
    return res

def test_mover_64_net(env, models):  # whole model
    from PIL import Image
    from utils.MCTS import MCT

    # HyperParameters
    gamma = 0.95
    actor = models[0]
    obj_num = 25
    action_space = 5
    action_size = obj_num * action_space
    batch_size = 1
    best_action_seq = []
    routes = []

    h = torch.zeros(1, 512).to(device)
    c = torch.zeros(1, 512).to(device)
    pre_action = np.zeros([1, action_size])


    init_state = (h,c)

    frames = []
    last_list = []

    time_dict = {}

    def start(id_=0):
        time_dict[id_] = time.time()

    def end(id_=0):
        return time.time() - time_dict[id_]

    finished_test = False
    total_reward = 0
    s = 0
    input_map = deepcopy(env.getmap())
    target_map = env.gettargetmap()

    tree = MCT()
    tree.root.state = deepcopy(env)
    t = time.time()
    tree.setactionsize(action_size)
    cflag = False

    tree.root.h_state = init_state  # which represents the last hidden state

    length = 0

    frame_cnt = 0
    while (not (finished_test == 1)) and s < 50:
        s += 1
        max_depth = 0
        h_state = tree.root.h_state

        state = env.getstate_1()
        tag = env.getfinished()
        _, h,c = actor(torch.FloatTensor(state).unsqueeze(0).to(device), torch.FloatTensor(pre_action).to(device),
                                       torch.FloatTensor(tag).unsqueeze(0).to(device), *h_state)
        h_state = (h,c)

        for i in range(action_size):
            if not tree.haschild(tree.root.id, i):
                item = int(i / action_space)
                direction = i % action_space
                env_copy = deepcopy(env)
                reward, done = env_copy.move(item, direction)
                succ, child_id = tree.expansion(tree.root.id, i, reward, env_copy, done)

                if succ:

                    # simulation
                    if done != 1:
                        policy = 0
                        cnt = 0
                        reward_sum_a = 0
                        env_sim = deepcopy(env_copy)
                        end_flag = False
                        h_state_t = h_state
                        action = i
                        while (not (end_flag == 1)) and cnt < 20:
                            cnt += 1

                            state = env_sim.getstate_1()
                            tag = env_sim.getfinished()
                            pre_action = np.zeros([1, action_size])
                            pre_action[0, action] = 1

                            logits,h,c  = actor(torch.FloatTensor(state).unsqueeze(0).to(device),
                                                           torch.FloatTensor(pre_action).to(device),
                                                           torch.FloatTensor(tag).unsqueeze(0).to(device),*h_state_t)
                            h_state_t = (h,c)

                            logits = logits.squeeze().detach().cpu().numpy()

                            while True:
                                action = np.argmax(logits)
                                item = int(action / action_space)
                                direction = action % action_space
                                reward, done = env_sim.move(item, direction)

                                if done != -1:
                                    break

                                logits[action] = -np.inf

                            reward_sum_a += reward * pow(gamma, cnt - 1)
                            end_flag = done

                        reward_sum = reward_sum_a
                        policy = 0

                        tree.nodedict[child_id].value = reward_sum
                        tree.nodedict[child_id].policy = policy
                        tree.nodedict[child_id].h_state = h_state

                        if reward_sum > 60:
                            print('wa done!', reward_sum, 'policy', policy)
                            cflag = True

                    tree.backpropagation(child_id)

        if cflag:
            C = 0.1
        else:
            C = 0.2

        cflag = False

        node_cnt = 0
        t_io = 0
        t_nn = 0
        t_tree = 0
        t_copy = 0
        start('total')
        action = 4

        while node_cnt < 200:
            start('tree')
            _id = tree.selection(C)
            t_tree += end('tree')
            policy = tree.nodedict[_id].policy
            start('copy')
            env_copy = deepcopy(tree.getstate(_id))
            t_copy += end('copy')
            h_state = tree.nodedict[_id].h_state

            if policy == 0:
                start('io')
                state = env_copy.getstate_1()
                tag = env_copy.getfinished()
                t_io += end('io')

                pre_action = np.zeros([1, action_size])
                pre_action[0, action] = 1

                start('nn')
                logits, h,c = actor(torch.FloatTensor(state).unsqueeze(0).to(device),
                                               torch.FloatTensor(pre_action).to(device),
                                               torch.FloatTensor(tag).unsqueeze(0).to(device), *h_state)

                t_nn += end('nn')

                h_state = (h,c)
                logits = logits.squeeze().detach().cpu().numpy()

            start('tree')
            empty_actions = tree.getemptyactions(_id)
            t_tree += end('tree')
            while True:

                action = np.argmax(logits)
                if not action in empty_actions:
                    logits[action] = -np.inf
                    continue

                start('io')
                item = int(action / action_space)
                direction = action % action_space
                reward, done = env_copy.move(item, direction)
                t_io += end('io')

                start('tree')
                succ, child_id = tree.expansion(_id, action, reward, env_copy, done)
                t_tree += end('tree')
                break

            # simulation
            if succ:
                node_cnt += 1
                if done != 1:

                    cnt = 0
                    reward_sum_a = 0
                    start('copy')
                    env_sim = deepcopy(env_copy)
                    t_copy += end('copy')
                    end_flag = False
                    h_state_t = h_state

                    while (not (end_flag == 1)) and cnt < 20:
                        cnt += 1

                        start('io')
                        state = env_sim.getstate_1()
                        tag = env_sim.getfinished()
                        t_io += end('io')

                        pre_action = np.zeros([1, action_size])
                        pre_action[0, action] = 1

                        start('nn')
                        logits, h,c = actor(torch.FloatTensor(state).unsqueeze(0).to(device),
                                                       torch.FloatTensor(pre_action).to(device),
                                                       torch.FloatTensor(tag).unsqueeze(0).to(device), *h_state_t)
                        t_nn += end('nn')
                        h_state_t = (h,c)

                        logits = logits.squeeze().detach().cpu().numpy()

                        while True:
                            action = np.argmax(logits)
                            item = int(action / action_space)
                            direction = action % action_space
                            start('io')
                            reward, done = env_sim.move(item, direction)
                            t_io += end('io')

                            if done != -1:
                                break

                            logits[action] = -np.inf

                        reward_sum_a += reward * pow(gamma, cnt - 1)
                        end_flag = done

                    reward_sum = reward_sum_a
                    policy = 0

                    tree.nodedict[child_id].value = reward_sum
                    tree.nodedict[child_id].policy = policy
                    tree.nodedict[child_id].h_state = h_state

                    if reward_sum > 60:
                        print('wa done!', reward_sum, 'policy', policy)
                        cflag = True

                max_depth = max([max_depth, tree.nodedict[child_id].depth])
                start('tree')
                tree.backpropagation(child_id)
                t_tree += end('tree')

        t_total = end('total')
        action = tree.root.best
        print(action)
        best_action_seq.append(action)
        item = int(action / action_space)
        direction = action % action_space

        prestate = env.cstate[item]
        prepos = env.pos[item]

        if direction != 4:
            spos = deepcopy(env.pos[item]).tolist()

        reward, done = env.move(item, direction)
        finished_test = done

        if direction == 4:
            length += route_length(env.getlastroute())

            routes.append(env.getlastroute())
        else:
            length += env.last_steps

            tpos = deepcopy(env.pos[item]).tolist()
            routes.append([spos,tpos])

        last_list = []
        node = tree.root
        best = node.best
        while best != -1:
            node = tree.nodedict[node.childs[best]]
            best = node.best
            last_list.append(node.id)

        tree.nextstep(action)
        total_reward += reward

    if finished_test == 1:
        print('Finished! Reward:', total_reward, 'Steps:', s, 'TL', length)
    else:
        print('Failed Reward:', total_reward, 'Steps:', s, 'TL', length)


    return best_action_seq, routes


def image2env(img, target):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)

    # invers_dist = np.ones_like(dst)*255 - dst
    invers_dist = dst


    kernel = np.ones((3, 3), np.uint8)
    invers_dist = cv2.dilate(invers_dist, kernel, iterations=1)
    # invers_dist = cv2.erode(invers_dist, kernel, iterations=1)

    binary,contours, hierarchy = cv2.findContours(invers_dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(dst,contours,-1,(120,0,0),2)

    count = 0
    single_object = []
    shapes = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    centers = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
               [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    color_list = []

    for cont in contours:
        cimg = np.ones_like(img) * 255
        # cimg = np.zeros_like(img)

        cv2.drawContours(cimg, contours, count, color=(0, 0, 0), thickness=-1)
        single_object.append(cimg)

        pts = np.where(cimg == 0)
        centers[count] = [int(sum(pts[0]) / len(pts[0])), int(sum(pts[1]) / len(pts[1]))]
        ptss = []
        for i in range(len(pts[0])):
            ptss.append((pts[0][i], pts[1][i]))
        shapes[count] = ptss

        x,y,w,h = cv2.boundingRect(cont)
        # print("x:{} y:{}".format(rect[0],rect[1]))
        # cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 0xff), 2)
        # y = 10 if rect[1] < 10 else rect[1]
        y = 10 if y < 10 else y

        # cv2.putText(img, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(img, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
        count += 1

    pos = centers
    print(pos)
    # target = [[58, 22], [57, 11],[55, 48], [48, 31],[41, 48] ,[40, 25] , [40, 13],[27, 47] , [25, 55], [18, 41], [16, 24], [15, 16],[16, 6] ,[15, 56] , [13, 48], [2, 44],[4, 27], [15, 16], [4, 3], [2, 56],[0, 0] ,[0, 0] , [0, 0], [0, 0], [0, 0], [0, 0]]
    # 10
    # target = [[57, 36], [57, 20], [53, 7], [57, 55], [43, 53], [24, 51], [24, 35], [43, 29], [30, 15],[14, 15], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    # 5
    # target = [[53, 26], [40, 33], [14, 57], [53, 43], [10, 13], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    print(target)
    shape = shapes
    cstate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tstate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wall = []
    env = ENV(size=(64, 64))
    env.setmap(pos, target, shape, cstate, tstate, wall)

    # visualize
    startmap = env.getmap()

    colors = [[225, 225, 225], [0, 0, 0], [78, 113, 190], [184, 90, 154], [156, 33, 137], [158, 2, 109], [184, 40, 138],
              [204, 47, 105], [143, 29, 120], [186, 135, 76], [233, 174, 106], [254, 227, 136], [255, 254, 160],
              [255, 233, 87], [242, 259, 63], [242, 117, 63], [232, 126, 81], [222, 140, 104], [1, 25, 53], [0, 52, 63],
              [29, 176, 184], [55, 198, 192], [208, 233, 255], [88, 131, 52], [74, 102, 165], [209, 168, 69],
              [188, 113, 61], [44, 181, 73], [40, 136, 255], [255, 160, 192], [155, 150, 219]]
    margin_width = 0
    grid_width = 10
    map_size = startmap.shape
    img_size = [map_size[0] * (grid_width + margin_width) + margin_width,
                map_size[1] * (grid_width + margin_width) + margin_width]
    img = np.ones([640,640, 3])
    img *= 255

    for i in range(map_size[0]):
        for j in range(map_size[1]):
            id = int(startmap[i, j])
            color = colors[id]
            bx = i * (grid_width + margin_width) + margin_width
            by = j * (grid_width + margin_width) + margin_width
            for k in range(grid_width):
                for l in range(grid_width):
                    img[bx + k, by + l] = color

    img = PImage.fromarray(np.uint8(img))
    # img.save('/home/baifan/catkin_ws/src/IPM_kinova_push_sim/fig3.png')

    return env, centers, contours



def move_to_joint(j0,j1,j2,j3,j4,j5):
    joint_goal = group.get_current_joint_values()
    joint_goal[0] = j0
    joint_goal[1] = j1
    joint_goal[2] = j2
    joint_goal[3] = j3
    joint_goal[4] = j4
    joint_goal[5] = j5
    group.go(joint_goal, wait=True)
    rospy.sleep(1)

def move_to_pose(pose):
    # Wrapper for move to position.
    p = pose.position
    o = pose.orientation
    move_to_position(p.x, p.y, p.z, o.x, o.y, o.z, o.w)

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

    group.set_start_state_to_current_state()
    group.set_pose_target(target_pose, end_effector_link)
    plan_success, traj, planning_time, error_code = group.plan()
    group.execute(traj)
    rospy.sleep(1)
    group.go()
    rospy.sleep(1)

def move_to_cpos(pose):
    wps = []
    wps.append(pose)
    group.limit_max_cartesian_link_speed(0.02, end_effector)
    (plan, fraction) = group.compute_cartesian_path(
            wps,
            0.01,
            0.0,
            True)

    group.execute(plan)
    group.clear_max_cartesian_link_speed()
    rospy.sleep(1)



if __name__ == '__main__':

    # ros
    bridge = CvBridge()
    rospy.init_node('ipm_detection',anonymous=True)

    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander('manipulator')

    goal = control_msgs.msg.GripperCommandGoal()
    robot_commander = moveit_commander.RobotCommander()
    end_effector_link = group.get_end_effector_link()
    reference_frame = 'base_link'
    group.set_pose_reference_frame(reference_frame)

    group.allow_replanning(True)

    group.set_goal_position_tolerance(0.01)
    group.set_goal_orientation_tolerance(0.01)

    group.set_max_acceleration_scaling_factor(0.03)
    group.set_max_velocity_scaling_factor(0.03)

    end_effector = group.get_end_effector_link()

    # Get the camera parameters
    print('waiting for camera information message')
    camera_info_msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
    K = camera_info_msg.K

    fx = K[0]
    cx = K[2]
    fy = K[4]
    cy = K[5]

    imitate_models_path = '/home/baifan/catkin_ws/src/IPM_kinova_push_sim/scripts/RL_models/80000-model.pkl'
    rl_models_path = '/home/baifan/catkin_ws/src/IPM_kinova_push_sim/scripts/RL_models/RL_IL3000/80000-model.pth'

    load_imitate_models = False
    load_rl_models = True

    state_size = [64, 64, 51]
    hidden_size = 512
    feature_size = 4096
    max_num = 25
    action_type = 5
    action_space = action_type * max_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N_worker = 1
    actor = Actor(state_size, hidden_size, max_num, action_type, feature_size).to(device)
    critic = Critic(hidden_size=hidden_size).to(device)

    models = [actor, critic]

    if load_imitate_models:
        states = torch.load(imitate_models_path)
        print("We load the imitate model from:",imitate_models_path)
        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)

        all_tuple = [("actor", actor)]
        for param in all_tuple:
            recover_state(*param)

    if load_rl_models:
        states = torch.load(rl_models_path)
        print("We load the rl model from:",rl_models_path)
        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)

        all_tuple = [("actor", actor), ("critic", critic)]
        for param in all_tuple:
            recover_state(*param)

    move_to_joint(2.85*np.pi/180, -99.49*np.pi/180, -47.57*np.pi/180, -122.96*np.pi/180, 90*np.pi/180, 3.28*np.pi/180)
    rospy.sleep(10)
    print('OK')

    color_message = rospy.wait_for_message("/camera/color/image_raw", Image)
    img = bridge.imgmsg_to_cv2(color_message,'bgr8')
    global depth
    depth_message = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
    depth = bridge.imgmsg_to_cv2(depth_message,'32FC1')
    global trans
    trans = get_trans('camera_color_optical_frame', 'base')
    # Crop a square out of the middle of the depth and resize it to 480 * 480
    crop_size = 480
    img_crop = cv2.resize(img[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size], (64, 64))

    # define the rearrangement target
    target = [[53, 26], [40, 33], [14, 57], [53, 43], [10, 13], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    env, pos, contours = image2env(img_crop, target)

    best_action_seq, routes = test_mover_64_net(env, models)
    print(best_action_seq)

    move_to_joint(1.1*np.pi/180, -106.6*np.pi/180, -102.35*np.pi/180, -61.09*np.pi/180, 90.08*np.pi/180, 1.69*np.pi/180)
    rospy.sleep(1)

    count = 0
    for i, val in enumerate(best_action_seq):
        # print i
        # print val
        # 0:up, 1:down, 2:left, 3:right
        # item = int(val / action_type)
        # act = int(val % action_type)

        path = routes[i]

        # start_point
        point2 = [path[0][0], path[0][1]]
        center_pixel = (
                    (np.array(point2) / 64.0 * crop_size) + np.array([(480 - crop_size) // 2, (640 - crop_size) // 2]))
        center_pixel = np.round(center_pixel).astype(np.int)
        point_depth = float(depth[center_pixel[0], center_pixel[1]])
        z = point_depth
        # These magic numbers are my camera intrinsic parameters.
        x = (center_pixel[1] - cx) / (fx) * point_depth
        y = (center_pixel[0] - cy) / (fy) * point_depth

        # Execute a grasp.
        tp = Pose()
        tp.position.x = x / 1000.0
        tp.position.y = y / 1000.0
        tp.position.z = z / 1000.0
        tp.orientation.w = 1

        tp_b = trans_pose(tp, trans, 'camera_color_optical_frame')

        # q = tft.quaternion_from_euler(np.pi, 0, theta)
        # print(tp_base)
        tp_base = Pose()
        tp_base.orientation.x = -0.71081157955
        tp_base.orientation.y = -0.703382256454
        tp_base.orientation.z = -0.000531853163616
        tp_base.orientation.w = 0.000129672712242

        tp_base.position.x = -tp_b.position.x
        tp_base.position.y = -tp_b.position.y
        tp_base.position.z = 0.16

        move_to_cpos(copy.deepcopy(tp_base))
        rospy.sleep(2)
        tp_base.position.z = 0.119
        move_to_cpos(copy.deepcopy(tp_base))
        rospy.sleep(2)

        waypoints = []
        # start with the current pose
        # waypoints.append(group.get_current_pose().pose)
        for i, point in enumerate(path):
            if i == 0:
                continue
            point2 = [point[0], point[1]]
            center_pixel = ((np.array(point2) / 64.0 * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
            center_pixel = np.round(center_pixel).astype(np.int)
            point_depth = float(depth[center_pixel[0], center_pixel[1]])
            z = point_depth
            # These magic numbers are my camera intrinsic parameters.
            x = (center_pixel[1] - cx)/(fx) * point_depth
            y = (center_pixel[0] - cy)/(fy) * point_depth

            # Execute a grasp.
            tp = Pose()
            tp.position.x = x/1000.0
            tp.position.y = y/1000.0
            tp.position.z = z/1000.0
            tp.orientation.w = 1


            tp_base = trans_pose(tp, trans, 'camera_color_optical_frame')

            # q = tft.quaternion_from_euler(np.pi, 0, theta)
            # print(tp_base)
            tp_base.orientation.x = -0.71081157955
            tp_base.orientation.y = -0.703382256454
            tp_base.orientation.z = -0.000531853163616
            tp_base.orientation.w = 0.000129672712242

            tp_base.position.x = -tp_base.position.x
            tp_base.position.y = -tp_base.position.y
            tp_base.position.z = 0.119
            waypoints.append(copy.deepcopy(tp_base))

        fraction = 0.0
        maxtries = 100
        attempts = 0

        group.set_start_state_to_current_state()

        while fraction < 1.0 and attempts < maxtries:
            group.limit_max_cartesian_link_speed(0.02, end_effector)
            (plan, fraction) = group.compute_cartesian_path(
                waypoints,
                0.01,
                0.0,
                True)

            attempts += 1

            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")

        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")
            group.execute(plan)
            group.clear_max_cartesian_link_speed()
            rospy.loginfo("Path execution complete.")
        else:
            rospy.loginfo("Path planning failed with only " + str(fraction) + " success after " + str(
                maxtries) + " attempts.")

        rospy.sleep(2)
        finish_point = group.get_current_pose().pose
        finish_point.position.z = 0.16
        move_to_cpos(copy.deepcopy(finish_point))

        rospy.sleep(1)

    move_to_joint(2.85*np.pi/180, -99.49*np.pi/180, -47.57*np.pi/180, -122.96*np.pi/180, 90*np.pi/180, 3.28*np.pi/180)

    rospy.sleep(1)
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)

