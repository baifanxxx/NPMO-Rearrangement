#import tensorflow as tf
import random
import os
import numpy as np
from copy import deepcopy
import time
from models import Actor,Critic
import torch
from utils.dataio import DataIO
import pickle
from torch.distributions import Categorical
from utils.env import ENV


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Random seed
manual_seed = 3
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

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

def test_mover_64_net(env, models,action_type,max_num):  # whole model
    from utils.MCTS import MCT


    # HyperParameters
    gamma = 0.95
    actor = models[0]
    obj_num = max_num
    action_space = action_type
    action_size = obj_num * action_space

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
                            logits = logits.squeeze()
                            while True:
                                dist = Categorical(logits=logits)
                                action = dist.sample()
                                action = action.item()
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
        print(C)

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

                logits = logits.squeeze()

            start('tree')
            empty_actions = tree.getemptyactions(_id)
            t_tree += end('tree')

            while True:
                dist = Categorical(logits=logits)
                action = dist.sample()
                action = action.item()

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
                        logits = logits.squeeze()

                        while True:
                            dist = Categorical(logits=logits)
                            action = dist.sample()
                            action = action.item()

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
        item = int(action / action_space)
        direction = action % action_space
        prestate = env.cstate[item]
        prepos = env.pos[item]
        reward, done = env.move(item, direction)
        finished_test = done
        if direction == 4:
            length += route_length(env.getlastroute())
        else:
            length += env.last_steps

        print('step', s, 'times', tree.nodedict[tree.root.childs[action]].times, 'id', tree.root.id, 'max_depth',
              max_depth - tree.root.depth, 'value', tree.root.value, 'best depth', tree.getbestdepth(), 'io time', t_io,
              'nn time', t_nn, 'copy time', t_copy, 'tree time', t_tree, 'total time', t_total, 'policy',
              tree.root.policy, 'current reward', total_reward)

        last_list = []
        node = tree.root
        best = node.best
        print('this')
        while best != -1:
            node = tree.nodedict[node.childs[best]]
            best = node.best
            print('id', node.id, 'value', node.value, 'reward', node.reward, 'best', best, 'depth',
                  node.depth - tree.root.depth)
            last_list.append(node.id)

        tree.nextstep(action)
        total_reward += reward

    if finished_test == 1:
        print('Finished! Reward:', total_reward, 'Steps:', s, 'TL', length)
    else:
        print('Failed Reward:', total_reward, 'Steps:', s, 'TL', length)

    return finished_test,total_reward,s,length



if __name__ == '__main__':

    # HyperParameters
    imitate_models_path = 'imitate_models/3000-model.pth'
    rl_models_path = 'RL_models/IL3000/80000-model.pth'

    load_imitate_models = False
    load_rl_models = True

    state_size = [64, 64, 51]
    hidden_size = 512
    feature_size = 4096
    max_num = 25
    action_type = 5
    action_space = action_type * max_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N_worker = 10
    actor = Actor(state_size, hidden_size, 25, action_type, feature_size).to(device)
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


    mean_total_reward = 0
    mean_s = 0
    mean_length = 0
    success_rate = 0
    mean_nn_time = 0

    #test envs
    envs = DataIO('/home/baifan/Projects/IPM_kinova_push_sim/data/test_20_2.pkl', N_worker)
    _ = envs.reset()

    total_reward_list = []
    s_list = []
    length_list = []
    finished_list = []
    time_list = []

    for i in range(N_worker):
        print(i)
        t_start = time.time()

        # test envs
        env = envs.envs[i]
        finished_test,total_reward,s,length = test_mover_64_net(env, models,action_type,max_num)
        t_end = time.time()

        total_reward_list.append(total_reward)
        s_list.append(s)
        length_list.append(length)
        finished_list.append(finished_test)
        time_list.append(t_end-t_start)

        mean_total_reward += total_reward
        mean_s += s
        mean_length += length
        success_rate += finished_test
        mean_nn_time += (t_end-t_start)

    dict_data = {"total_reward":total_reward_list,"step":s_list,"length":length_list,"finished":finished_list,"time":time_list}

    print('Finished Test! mean_total_reward:', mean_total_reward/float(N_worker), ' mean_s:', mean_s/float(N_worker), ' mean_length:', mean_length/float(N_worker),' success_rate:',success_rate/float(N_worker),' mean_nn_time:',mean_nn_time/float(N_worker))

    # with open("/home/baifan/Projects/IPM_kinova_push_sim/scripts/test_data_20_RL_models_IL3000.pkl", 'wb') as fo:
    #     pickle.dump(dict_data, fo)
