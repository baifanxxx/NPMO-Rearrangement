import torch.nn.functional as F
from torch.distributions import Categorical
from models import Actor,Critic
import os
import torch
import numpy as np
from utils.dataio import DataIO
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

tensorboard_path = 'RL_tensorboard/IL800/'
imitate_models_path = 'imitate_models/800-model.pth'
rl_models_path = 'RL_models/14400-model.pth'
weight_path = 'RL_models/IL800'
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

# HyperParameters
train_max_step = 80001
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
N_worker = 28

save_freq = 40000
test_freq = 5000

state_size = [64, 64, 51]
hidden_size = 512
feature_size = 4096
max_num = 25
action_type = 5
action_space = action_type * max_num


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
load_imitate_models = True
load_rl_models = False

actor = Actor(state_size, hidden_size, max_num, action_type, feature_size).to(device)
critic = Critic(hidden_size=hidden_size).to(device)
actor_opt = torch.optim.Adam(actor.parameters(), lr=lr,eps=eps)
critic_opt = torch.optim.Adam(critic.parameters(), lr=lr,eps=eps)
models = [actor, critic]
optimizers = [actor_opt, critic_opt]


writer = SummaryWriter(tensorboard_path)



def train(models, optimizers, states,pre_actions,tags, actions, rewards, dones, old_action_log_probs, final_state,final_pre_action,final_tag, start_h, start_c,step):
    actor = models[0]
    critic = models[1]

    states = torch.FloatTensor(states).to(device)  # (T, N, 1, 84, 84)
    pre_actions = torch.LongTensor(pre_actions).to(device)  # (T, N, 4)
    tags = torch.LongTensor(tags).to(device)  # (T, N, 4)

    actions = torch.LongTensor(actions).view(-1, 1).to(device)  # (T*N, 1)
    rewards = torch.FloatTensor(rewards).view(-1, 1).to(device)  # (T*N, 1)
    dones = torch.FloatTensor(dones).to(device)  # (T, N)
    old_action_log_probs = torch.FloatTensor(old_action_log_probs).view(-1, 1).to(device)  # (T*N, 1)
    final_state = torch.FloatTensor(final_state).to(device)
    final_pre_action = torch.FloatTensor(final_pre_action).to(device)
    final_tag = torch.FloatTensor(final_tag).to(device)

    mean_loss = 0.
    mean_actor_loss = 0.
    mean_critic_loss = 0.
    mean_entropy_loss = 0.

    for _ in range(K_epoch):
        # Calculate Probs, values
        probs = []
        values = []
        h = start_h
        c = start_c

        for state,pre_action, tag, done in zip(states,pre_actions, tags, dones):
            prob, h, c = actor(state, pre_action, tag, h, c)
            value = critic(h)

            probs.append(prob)
            values.append(value)
            for i, d in enumerate(done):
                if d.item() == 0:
                    h, c = reset_hidden(i, h, c)

        _, h, c = actor(final_state, final_pre_action, final_tag, h, c)
        final_value = critic(h)

        next_values = values[1:]
        next_values.append(final_value)

        probs = torch.cat(probs)  # (T*N, 4)

        values = torch.cat(values)  # (T*N, 1)
        next_values = torch.cat(next_values)  # (T*N, 1)

        td_targets = rewards + gamma * next_values * dones.view(-1, 1)  # (T*N, 1)
        deltas = td_targets - values  # (T*N, 1)

        # calculate GAE
        deltas = deltas.view(-1, N_worker, 1).cpu().detach().numpy()  # (T, N, 1)
        masks = dones.view(-1, N_worker, 1).cpu().numpy()
        advantages = []
        advantage = 0
        for delta, mask in zip(deltas[::-1], masks[::-1]):
            advantage = gamma * lambd * advantage * mask + delta
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.FloatTensor(advantages).view(-1, 1).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # new_action_probs = torch.prod(actions * probs + (1 - actions) * (1 - probs), dim=1, keepdim=True)  # (T*N, 1)
        dist = Categorical(logits=probs)

        new_action_log_probs = dist.log_prob(actions.view(-1)).view(-1, 1)

        ratio = torch.exp(new_action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages

        actor_loss = coef_act * (-torch.mean(torch.min(surr1, surr2)))
        critic_loss = coef_crit * F.smooth_l1_loss(values, td_targets.detach())

        dist_entropy = dist.entropy().mean()
        # entropys = torch.sum(-probs * torch.log(probs) + -(1 - probs) * torch.log(1 - probs), dim=1)
        entropy_loss = coef_entropy * dist_entropy

        loss = actor_loss + critic_loss + entropy_loss

        mean_loss += loss.item()
        mean_actor_loss += actor_loss.item()
        mean_critic_loss += critic_loss.item()
        mean_entropy_loss += entropy_loss.item()


        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(actor.parameters(), max_grad_norm)
        # nn.utils.clip_grad_norm(critic.parameters(), max_grad_norm)
        for optimizer in optimizers:
            optimizer.step()

    mean_loss = mean_loss/K_epoch
    mean_actor_loss = mean_actor_loss/K_epoch
    mean_critic_loss = mean_critic_loss/K_epoch
    mean_entropy_loss = mean_entropy_loss/K_epoch


    print('Step:',step,' mean_loss:', mean_loss, ' mean_actor_loss:', mean_actor_loss, ' mean_critic_loss:', mean_critic_loss,' mean_entopy_loss:', mean_entropy_loss)
    writer.add_scalar('loss', mean_loss, step)
    writer.add_scalar('actor_loss', mean_actor_loss, step)
    writer.add_scalar('critic_loss', mean_critic_loss, step)
    writer.add_scalar('entopy_loss', mean_entropy_loss, step)




def reset_hidden(i, h, c):
    filter_tensor = torch.ones_like(h)
    filter_tensor[i] = torch.zeros(512)
    h = h * filter_tensor
    c = c * filter_tensor
    return h, c


def main():
    states = None
    if load_imitate_models:
        print("We load the imitate model from:",imitate_models_path)
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
        print("We load the rl model from:",rl_models_path)
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

    env = DataIO('/home/baifan/Projects/IPM_kinova_push_sim/data/train_IL2.pkl', N_worker,action_type=action_type)

    print("Train Start")
    while step <= train_max_step:
        _ = env.reset()
        state, tag = env.get_state()
        h = torch.zeros(N_worker, 512).to(device)
        c = torch.zeros(N_worker, 512).to(device)
        pre_action = np.zeros([N_worker, action_space])
        start_h = h
        start_c = c
        ended = np.array([False] * N_worker)
        states, pre_actions, tags, actions, rewards, dones, old_action_log_probs = list(), list(), list(), list(), list(), list(), list()
        for _ in range(T_horizon):
            logits, next_h, next_c = actor(torch.FloatTensor(state).to(device), torch.FloatTensor(pre_action).to(device), torch.FloatTensor(tag).to(device), h, c)

            reward_done, a_t_cpu, old_action_log_prob = env.make_action_logits(logits)
            next_state, next_tag = env.get_state()

            next_pre_action = np.zeros([N_worker, action_space])
            for i, a_ in enumerate(a_t_cpu):
                next_pre_action[i, a_] = 1

            old_action_log_prob = torch.FloatTensor(old_action_log_prob).view(-1,1).to(device)
            old_action_log_prob = old_action_log_prob.cpu().detach().numpy()

            action = a_t_cpu

            done = np.array([item[1] for item in reward_done])
            reward = np.array([item[0] for item in reward_done])


            # save transition
            states.append(state)  # (T, N, 1, 84, 84)
            pre_actions.append(pre_action)  # (T, N, 4)
            tags.append(tag)  # (T, N, 4)

            actions.append(action)  # (T, N, 1)
            rewards.append(reward)  # (T, N)
            dones.append(1 - done)  # (T, N)
            old_action_log_probs.append(old_action_log_prob)  # (T, N, 1)

            # record score and check done
            for i, (r, d) in enumerate(zip(reward, done)):
                if d == True:
                    next_h, next_c = reset_hidden(i, next_h, next_c)  # if done, reset hidden

            state = next_state
            pre_action = next_pre_action
            tag = next_tag
            h = next_h.detach()
            c = next_c.detach()

            step += 1

            ended = np.logical_or(ended, (done == 1))
            if ended.all():
                break

        print('Step:',step,' rewards:',sum(sum(rewards))/N_worker,' len:',len(rewards))
        writer.add_scalar('rewards', sum(sum(rewards))/N_worker, step)

        train(models, optimizers, states,pre_actions,tags, actions, rewards, dones, old_action_log_probs, state, pre_action, tag, start_h, start_c,step)


        #save model
        if step % save_freq == 0 and step != 0:
            states = {}
            def create_state(name, model, optimizer):
                states[name] = {
                    'time_step': step + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

            all_tuple = [("actor", actor, actor_opt),
                         ("critic", critic, critic_opt)]
            for param in all_tuple:
                create_state(*param)
            torch.save(states, weight_path + '/' + str(step) + '-model.pth',_use_new_zipfile_serialization=False)
            print('Save ', str(step) + '-model.pth')

        #test
        if step % test_freq == 0 and step != 0:
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

                    for ep in range(T_horizon):

                        logits, next_h, next_c = actor(torch.FloatTensor(state).to(device),
                                                       torch.FloatTensor(pre_action).to(device),
                                                       torch.FloatTensor(tag).to(device), h, c)

                        reward_done, a_t_cpu, old_action_log_prob = env.make_action_logits(logits)
                        next_state, next_tag = env.get_state()

                        next_pre_action = np.zeros([N_worker, action_space])
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
                    traj_length += T_horizon * (ended == 0)

                print('Test:', ' rewards:', cout_rewards / (batch_size*iters), ' success_rate:', endeds / (batch_size*iters),' average_step:', traj_length.mean())
                writer.add_scalar('test_rewards', cout_rewards / (batch_size*iters), step)
                writer.add_scalar('test_success_rate', endeds / (batch_size*iters), step)
                writer.add_scalar('test_average_step', traj_length.mean(), step)

    print("Train End !!!")


if __name__ == "__main__":
    main()