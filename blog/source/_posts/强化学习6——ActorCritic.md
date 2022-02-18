---
title: 强化学习6——ActorCritic
copyright: true
mathjax: true
date: 2021-03-27 23:04:46
tags:
categories: 强化学习
---

概述：首页描述

![]()

<!--more-->

## Actor Critic

&emsp;&emsp;虽然Poclicy Gradient成功解决了Value-base算法不能解决在连续动作中进行选择的问题，但是由于其只能进行回合制更新，因此具有更新效率低，算法稳定性差等原因，需要进一步完善，因此就发明了一种全新的算法——Actor Critic，这种算法将Value-base算法与Gradient算法进行结合，使Policy Gradient也能在每个step进行更新。

### 1. 基本情况

&emsp;&emsp;Actor Critic通过建立两个深度神经网络，Policy Gradient神经网络作为**Actor**，**基于概率进行行为选择**，而Value-base的神经网络作为**Critic，基于ACtor采取的行为评判行为得分**，**Actor根据Critic的评分修改选行为的概率。**

> **Critic: Value-base**，对每个action进行评价，指导Actor更新
>
> **Actor: Policy Gradient**，行动选择，根据Critic的评分修改选行为概率

**算法目标：改善Policy Gradient只能回合制更新导致的低效率问题**

**优势**：可以进行单步更新，比传统Policy Gradient速度更快

**缺点**：难收敛，由于Actor Critic取决于Critic对于价值的判断，但是Critic难以收敛，再加上Actor的更新，就更加难以收敛了。



### 2. 代码分析

#### 更新框架

> 1. 开始游戏，初始化state
> 2. Actor根据当前状态选择action
> 3. 与环境交互获得next_state、reward、done
> 4. Critic根据state、reward、next_state机选**td_error**，并重新训练Critic中的PNet。
> 5. Actor根据state、action、**td_error**进行重训练
> 6. state=next_state进行下一轮训练

对应代码如下：

~~~python
for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = actor.choose_action(state)  # SoftMax概率选择action
            next_state, reward, done, _ = env.step(action)
            td_error = critic.train_Q_network(state, reward, next_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            state = next_state
            if done:
                break
~~~

#### 行为选择

&emsp;&emsp;Actor Critic模型的行为选择由Actor进行，与一般地Policy Gradient完全一致。

~~~python
def choose_action(self, observation):
      	# 标准acotor行为选择
      	# 1. 根据观测值输入到actor网络中获得各个行为被选择的高绿softmax后概率值
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
      	# 2. 根据个行为被选择概率随机进行概率选择
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action
~~~

#### Critic更新与td_error生成

&emsp;&emsp;在Actor Critic中使用一种衡量”比平时好多少的“的指标`td-error`来当做reward，其计算公式为：

​								$$td\_error= reward+\gamma V(S^{'})-V(S)$$

> 其中S为当前状态，$S'$为执行action后下一状态，V函数表示Critic中Value-Base网络输出结果。

&emsp;&emsp;然后使用**td_error的方差**作为损失函数函数进行Critic的Value-Base网络进行更新。

```python
def train_Q_network(self, state, reward, next_state):
    s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
    # 前向传播
    v = self.network.forward(s)     # v(s)
    v_ = self.network.forward(s_)   # v(s')

    # 反向传播
    loss_q = self.loss_func(reward + GAMMA * v_, v)
    self.optimizer.zero_grad()
    loss_q.backward()
    self.optimizer.step()
		# td_error计算
    with torch.no_grad():
        td_error = reward + GAMMA * v_ - v

    return td_error
```

#### Actor更新

&emsp;&emsp;在Actor Critic中的梯度下降与标准的Policy Gradient模型类似，主要的区别在于两点:

1. 经典Policy Gradient由于是回合制进行更新的，进行模型更新使用的是state序列和action序列及其对应的概率，而Policy使用的单个action与state及其对应的概率值。 
2. Actor Critic使用Critic产生的`td-error`来进行交叉熵损失函数加权更新Policy model，而Policy Gradient中使用G值。

~~~python
def learn(self, state, action, td_error):
        self.time_step += 1
        # Step 1: 前向传播,获取交叉熵损失
        softmax_input = self.network.forward(torch.FloatTensor(state).to(device)).unsqueeze(0)
        action = torch.LongTensor([action]).to(device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')

        # Step 2: 反向传播
        # 这里需要最大化当前策略的价值，因此需要最大化neg_log_prob * tf_error,即最小化-neg_log_prob * td_error
        loss_a = -neg_log_prob * td_error		
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()
~~~

### 3. 完整代码

~~~python
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Hyper Parameters for Actor
GAMMA = 0.95  # discount factor
LR = 0.01  # learning rate

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  # 非确定性算法


class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Actor(object):
    # dqn Agent
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
      	# 标准acotor行为选择
      	# 1. 根据观测值输入到actor网络中获得各个行为被选择的高绿softmax后概率值
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
      	# 2. 根据个行为被选择概率随机进行概率选择
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    def learn(self, state, action, td_error):
        self.time_step += 1
        # Step 1: 前向传播,获取交叉熵损失
        softmax_input = self.network.forward(torch.FloatTensor(state).to(device)).unsqueeze(0)
        action = torch.LongTensor([action]).to(device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')

        # Step 2: 反向传播
        # 这里需要最大化当前策略的价值，因此需要最大化neg_log_prob * tf_error,即最小化-neg_log_prob * td_error
        loss_a = -neg_log_prob * td_error		
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()


# Hyper Parameters for Critic
EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)   # 这个地方和之前略有区别，输出不是动作维度，而是一维

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Critic(object):
    def __init__(self, env):
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # init network parameters
        self.network = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # init some parameters
        self.time_step = 0
        self.epsilon = EPSILON  # epsilon值是随机不断变小的

    def train_Q_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
        # 前向传播
        v = self.network.forward(s)     # v(s)
        v_ = self.network.forward(s_)   # v(s')

        # 反向传播
        loss_q = self.loss_func(reward + GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v

        return td_error


# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = actor.choose_action(state)  # SoftMax概率选择action
            next_state, reward, done, _ = env.step(action)
            td_error = critic.train_Q_network(state, reward, next_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is ', time_end - time_start, 's')
~~~












