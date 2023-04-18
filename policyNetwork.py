import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size=0):
        super(PolicyNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 128)
        #self.linear3 = nn.Linear(256, 256)
        if self.action_size > 0:
            self.linear3 = nn.Linear(128, action_size)
        else:
            self.linear3 = nn.Linear(128, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        #output = F.relu(self.linear3(output))
        output = self.linear3(output)
        if self.action_size > 0:
           return F.softmax(output, dim=-1)
        else:
            return output


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.obs_dim = state_size
        self.act_dim = action_size
        self.pi = PolicyNet(state_size, action_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state, batch_mode=False):

        state = torch.from_numpy(np.array(state)).float()
        if not batch_mode:
            state = state.unsqueeze(0)

        state = Variable(state).to(self.device)
        state = state.view(state.shape[0], -1)

        return self.pi(state)


class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.v = PolicyNet(state_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, batch_mode=False):
        state = Variable(torch.from_numpy(np.array(state)).float()).to(self.device)
        if not batch_mode:
            state = state.unsqueeze(0)
        state = state.view(state.shape[0], -1)
        v = self.v(state)
        return torch.squeeze(v, -1)


class ActorCriticNet(nn.Module):
    def __init__(
        self, state_size, action_size
    ):
        super().__init__()

        # build policy and value functions
        self.PI = Actor(state_size, action_size)
        self.V = Critic(state_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, batch_mode=False):
        return self.PI(state), self.V(state)

