import os
import torch
import torch.nn as nn
from models import Actor, Critic
import numpy as np
from collections import defaultdict
from utils.dataio import DataIO
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tensorboard_path = 'imitate_tensorboard'
weight_path = 'imitate_models'
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

# HyperParameters
epochs = 901
batch_size = 64
state_size = [64, 64, 51]
hidden_size = 512
feature_size = 4096
max_num = 25
action_type = 5
action_space = action_type * max_num
episode_len = 80
save_freq = 100
test_freq = 100
eps = 1e-5
lr = 1e-4
load = False


env = DataIO('/home/baifan/Projects/IPM_kinova_push_sim/data/train_IL2.pkl', batch_size)
actor = Actor(state_size, hidden_size, max_num, action_type, feature_size).to(device)
critic = Critic(hidden_size=hidden_size).to(device)
actor_opt = torch.optim.Adam(actor.parameters(), lr=lr,eps=eps)
critic_opt = torch.optim.Adam(critic.parameters(), lr=lr,eps=eps)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
models = [actor, critic]
optimizers = [actor_opt, critic_opt]

writer = SummaryWriter(tensorboard_path)

def teacher_action(t, actions):
    res = []
    for i, a in enumerate(actions):
        if t >= len(a):
            res.append(-1)
        else:
            it, d = a[t]
            res.append(it * action_type + d)

    return torch.from_numpy(np.array(res)).to(device)

def reset_hidden(i, h, c):
    filter_tensor = torch.ones_like(h)
    filter_tensor[i] = torch.zeros(512)
    h = h * filter_tensor
    c = c * filter_tensor
    return h, c


for epoch in range(epochs):
    logs = defaultdict(list)
    sumloss = 0.
    for model in models:
        model.train()

    for optim in optimizers:
        optim.zero_grad()

    batch_size = env.batch_size
    actions = env.reset()

    c1 = torch.from_numpy(np.zeros([batch_size, hidden_size])).float().to(device)
    h1 = torch.from_numpy(np.zeros([batch_size, hidden_size])).float().to(device)

    ended = np.array([False] * batch_size)

    pre_action = np.zeros([batch_size, action_space])
    pre_action = torch.from_numpy(pre_action).float().to(device)

    loss = 0
    cnt = 0
    for t in range(episode_len):
        state, tag = env.get_state()
        state = torch.from_numpy(state).float().to(device)
        tag = torch.from_numpy(tag).float().to(device)

        logits, h1, c1 = actor(state, pre_action, tag, h1, c1)

        teacher = teacher_action(t, actions)
        loss += (criterion(logits, teacher) * (1.0 - torch.from_numpy(ended).float().to(device))).sum()

        cnt += (1.0 - torch.from_numpy(ended).float().to(device)).sum().item()

        a_t_cpu = teacher.cpu().numpy()

        pre_action = np.zeros([batch_size, action_space])
        for i, a_ in enumerate(a_t_cpu):
            pre_action[i, a_] = 1
        pre_action = torch.from_numpy(pre_action).float().to(device)

        rewards = env.make_action(a_t_cpu)
        done = np.array([item[1] for item in rewards])

        ended = np.logical_or(ended, (done == 1))

        if ended.all():
            break
    # print(t)
    sumloss += 1 * loss / cnt
    logs['loss'] = sumloss.item()
    print('Epoch: ',str(epoch),' Loss: ',sumloss.item())
    writer.add_scalar('loss', sumloss.item(), epoch)


    sumloss.backward()

    for optim in optimizers:
        optim.step()

    if epoch % save_freq == 0 and epoch != 0:
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }


        all_tuple = [("actor", actor, actor_opt),
                     ("critic", critic, critic_opt)]

        for param in all_tuple:
            create_state(*param)
        torch.save(states, weight_path+'/'+str(epoch)+'-model.pth')
        print('Save ',str(epoch)+'-model.pth')

    if epoch % test_freq == 0 and epoch != 0:
        logs = defaultdict(list)
        env.idx = 0
        for model in models:
            model.eval()

        iters = 5
        cout_rewards = 0
        endeds = 0
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

                for ep in range(episode_len):

                    logits, next_h, next_c = actor(torch.FloatTensor(state).to(device),
                                                   torch.FloatTensor(pre_action).to(device),
                                                   torch.FloatTensor(tag).to(device), h, c)

                    reward_done, a_t_cpu, old_action_log_prob = env.make_action_logits(logits)
                    next_state, next_tag = env.get_state()

                    next_pre_action = np.zeros([batch_size, action_space])
                    for i, a_ in enumerate(a_t_cpu):
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
                traj_length += episode_len * (ended == 0)

            print('Test:', ' rewards:', cout_rewards / (batch_size * iters), ' success_rate:',
                  endeds / (batch_size * iters), ' average_step:', traj_length.mean())
            writer.add_scalar('test_rewards', cout_rewards / (batch_size * iters), epoch)
            writer.add_scalar('test_success_rate', endeds / (batch_size * iters), epoch)
            writer.add_scalar('test_average_step', traj_length.mean(), epoch)







