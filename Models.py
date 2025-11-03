import torch
import torch.nn as nn
from torch.nn import functional as F

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class RLAgent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=0.001, epsilon=0.1, gamma=0.99):
        super(RLAgent, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(self, state, training=True):
        if training and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self(state)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state):
        self.optimizer.zero_grad()
        q_values = self(state)
        q_value = q_values[action]
        with torch.no_grad():
            next_q_values = self(next_state)
            target = reward + self.gamma * next_q_values.max()
        loss = self.loss_fn(q_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class AGN(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(AGN, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_channels, 1))
        self.register_buffer('running_var', torch.zeros(1, num_channels, 1))
        self.rl_agent = RLAgent(state_dim=3, action_dim=2)  # State: [mean, var, loss/comm_round], Actions: [GN, BN]
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def compute_state(self, x, loss=None, comm_round=None):
        x_flat = x.view(x.size(0), x.size(1), -1)
        mean = x_flat.mean(-1).mean(0).mean()  # Scalar mean
        var = x_flat.var(-1).mean(0).mean()    # Scalar variance
        state = [mean, var]
        if loss is not None:
            state.append(loss)
        elif comm_round is not None:
            state.append(comm_round / 200.0)  # Normalize comm_round
        else:
            state.append(0.0)
        return torch.tensor(state, dtype=torch.float32, device=x.device)

    def forward(self, x, loss=None, comm_round=None):
        x_group = x.view(1, x.size(0) * self.num_groups, -1)
        N, C, H, W = x.size()
        G = self.num_groups
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        temp = var_in + mean_in ** 2

        mean_gn = x_group.mean(-1, keepdim=True)
        var_gn = x_group.var(-1, keepdim=True)
        mean_gn = mean_gn.view(x.size(0), G, -1)
        var_gn = var_gn.view(x.size(0), G, -1)
        mean_gn = torch.stack([mean_gn] * (C // G), dim=2).view(x.size(0), C, -1)
        var_gn = torch.stack([var_gn] * (C // G), dim=2).view(x.size(0), C, -1)

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        state = self.compute_state(x, loss, comm_round)
        action = self.rl_agent.select_action(state, self.training)
        mean = mean_gn if action == 0 else mean_bn
        var = var_gn if action == 0 else var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        output = x * self.weight + self.bias

        return output, state, action

    def update_rl(self, state, action, reward, next_state):
        return self.rl_agent.update(state, action, reward, next_state)

class client_model(nn.Module):
    def __init__(self, name):
        super(client_model, self).__init__()
        self.name = name
        if 'LeNet' in self.name:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, 10)
            if self.name.endswith('_bn'):
                self.bn1 = nn.BatchNorm2d(64)
                self.bn2 = nn.BatchNorm2d(64)
            elif self.name.endswith('_gn'):
                self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
                self.bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            elif self.name.endswith('_fednn'):
                self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=5)
                self.conv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=5)
                self.bn1 = AGN(num_groups=2, num_channels=64)
                self.bn2 = AGN(num_groups=2, num_channels=64)
        else:
            raise ValueError()

    def forward(self, x, loss=None, comm_round=None):
        if 'LeNet' in self.name:
            if self.name == 'LeNet':
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
            else:
                x, state1, action1 = self.bn1(self.conv1(x), loss, comm_round)
                x = self.pool(F.relu(x))
                x, state2, action2 = self.bn2(self.conv2(x), loss, comm_round)
                x = self.pool(F.relu(x))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            if self.name.endswith('_fednn') and (state1 is not None or state2 is not None):
                return x, [(state1, action1), (state2, action2)]
        return x