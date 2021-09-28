import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size=[64, 64, 51], hidden_size=512, max_num=25, action_type=5, feature_size=4096):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_space = max_num * action_type
        self.max_num = max_num
        self.action_type = action_type

        self.flat = nn.Flatten()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels=state_size[-1],
                               out_channels=64,
                               kernel_size=5,
                               stride=2,
                               padding=2)

        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.resconv = nn.Conv2d(in_channels=64,
                               out_channels=256,
                               kernel_size=1,
                               stride=4)
        self.lstm = nn.LSTMCell(feature_size//2, hidden_size)
        self.linear = nn.Linear(hidden_size, self.action_space, bias=False)
        self.fusionlinear = nn.Linear(feature_size + self.action_space + self.max_num, feature_size//2, bias=False)


    def forward(self, x, pre_action, finish_tag, h_0, c_0):
        x = x.float()
        x = x.permute(0, 3, 2, 1)
        c1 = self.lrelu(self.conv1(x))  # 32 x 32 x 64
        res = c1
        c2 = self.lrelu(self.conv2(c1))  # 16 x 16 x 128
        c3 = self.lrelu(self.conv3(c2))  # 8  x  8 x 256
        c3 = c3 + self.lrelu(self.resconv(res))
        c4 = self.conv4(c3)  # 4  x  4 x 256
        flat = self.flat(c4)  # 4096
        feature = torch.cat([flat, pre_action.float(), finish_tag.float()], -1)
        feature = self.fusionlinear(feature)
        h_1, c_1 = self.lstm(feature, (h_0, c_0))
        logits = self.linear(h_1)
        return logits, h_1, c_1


class Critic(nn.Module):
    def __init__(self, hidden_size=512):
        super(Critic, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        x = self.lrelu(self.linear1(x))
        res = self.linear2(x)

        return res

