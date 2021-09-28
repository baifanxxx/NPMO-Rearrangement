import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import time
from models import Actor,Critic

import os
import torch
import numpy as np
from utils.dataio import DataIO


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

tensorboard_path = 'RL_tensorboard'
imitate_models_path = 'imitate_models/3000-model.pth'
rl_models_path = 'RL_models/IL3000/80000-model.pth'
weight_path = 'RL_models'
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

# Random seed 3
manual_seed = 3
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

# HyperParameters
train_max_step = 30000000
eps_clip = 0.1
gamma = 0.95
lambd = 0.95
lr = 2e-4
eps = 1e-5
coef_act = 1
coef_crit = 0.5
coef_entropy = 0.001

max_grad_norm = 0.5

T_horizon = 50
K_epoch = 8
N_worker = 1

save_interval = 10
save_freq = 1000
test_freq = 1000

state_size = [64, 64, 51]
hidden_size = 512
feature_size = 4096
max_num = 25
action_type = 5
action_space = action_type * max_num


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
load_imitate_models = False
load_rl_models = True

actor = Actor(state_size, hidden_size, max_num, action_type, feature_size).to(device)
critic = Critic(hidden_size=hidden_size).to(device)
actor_opt = torch.optim.Adam(actor.parameters(), lr=lr,eps=eps)
critic_opt = torch.optim.Adam(critic.parameters(), lr=lr,eps=eps)
models = [actor, critic]
optimizers = [actor_opt, critic_opt]


def reset_hidden(i, h, c):
    filter_tensor = torch.ones_like(h)
    filter_tensor[i] = torch.zeros(512)
    h = h * filter_tensor
    c = c * filter_tensor
    return h, c

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


states = None
if load_imitate_models:
    print("We load the imitate model from:", imitate_models_path)
    states = torch.load(imitate_models_path)
    def recover_state(name, model, optimizer):
        state = model.state_dict()
        model_keys = set(state.keys())
        load_keys = set(states[name]['state_dict'].keys())
        if model_keys != load_keys:
            print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
        state.update(states[name]['state_dict'])
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]['optimizer'])
    all_tuple = [("actor", actor, actor_opt)]
    for param in all_tuple:
        recover_state(*param)

if load_rl_models:
    print("We load the rl model from:", rl_models_path)
    states = torch.load(rl_models_path)
    def recover_state(name, model, optimizer):
        state = model.state_dict()
        model_keys = set(state.keys())
        load_keys = set(states[name]['state_dict'].keys())
        if model_keys != load_keys:
            print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
        state.update(states[name]['state_dict'])
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]['optimizer'])
    all_tuple = [("actor", actor, actor_opt), ("critic", critic, critic_opt)]
    for param in all_tuple:
        recover_state(*param)

if load_rl_models and states:
    step = states["critic"]['time_step']
else:
    step = 0

env = DataIO('/home/baifan/Projects/IPM_kinova_push_sim/data/test_20_2.pkl', N_worker)

env.idx = 0
for model in models:
    model.eval()

iters = 10
cout_rewards = 0
cout_step = 0
endeds = 0
length = 0
t_start = time.time()
with torch.no_grad():
    for i in range(iters):
        batch_size = env.batch_size
        _ = env.reset()
        state, tag = env.get_state()
        h = torch.zeros(batch_size, 512).to(device)
        c = torch.zeros(batch_size, 512).to(device)
        pre_action = np.zeros([batch_size, action_space])
        ended = np.array([False] * batch_size)

        traj_length = np.zeros(batch_size).astype(np.int32)

        for ep in range(T_horizon):

            logits, next_h, next_c = actor(torch.FloatTensor(state).to(device),
                                           torch.FloatTensor(pre_action).to(device),
                                           torch.FloatTensor(tag).to(device), h, c)

            reward_done = []
            actions = []
            action_log_probs = []
            for i, lo in enumerate(logits):
                while True:

                    dist = Categorical(logits=lo)
                    action = dist.sample()
                    a = action.item()
                    # action = torch.argmax(lo)
                    # a = action.item()

                    item = int(a / action_type)
                    d = int(a % action_type)
                    reward, flag = env.envs[i].move(item, d)
                    if flag == -1:
                        lo[a] = -np.inf
                        continue
                    action_log_probs.append(dist.log_prob(action))
                    reward_done.append((reward, flag))
                    actions.append(a)
                    break
            direction = d
            if direction == 4:
                length += route_length(env.envs[0].getlastroute())
            else:
                length += env.envs[0].last_steps

            next_state, next_tag = env.get_state()

            next_pre_action = np.zeros([N_worker, action_space])
            for i, a_ in enumerate(actions):
                next_pre_action[i, a_] = 1

            done = np.array([item[1] for item in reward_done])
            reward = np.array([item[0] for item in reward_done])

            traj_length += (ep + 1) * np.logical_and(ended == 0, (done == 1))

            ended = np.logical_or(ended, (done == 1))

            # record score and check done
            for i, (r, d) in enumerate(zip(reward, done)):
                cout_rewards += r
                if d == True:
                    next_h, next_c = reset_hidden(i, next_h, next_c)  # if done, reset hidden

            state = next_state
            pre_action = next_pre_action
            tag = next_tag
            h = next_h.detach()
            c = next_c.detach()

            if ended.all():
                break

        endeds += ended.sum()
        traj_length += T_horizon * (ended == 0)

        cout_step += traj_length[0]
    t_end = time.time()
    print('Test:', ' rewards:', cout_rewards / float(batch_size * iters), ' success_rate:',
          endeds / float(batch_size * iters), ' average_step:', cout_step/float(batch_size * iters), ' mean_length:', length/float(batch_size * iters), ' mean_nn_time:',(t_end-t_start)/float(batch_size * iters))