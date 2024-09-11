import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy                                   # 导入gym
import sys
# 超参数
SCALE = 5000
BATCH_SIZE = 360                                # 样本数量
LR = 0.00025                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.95                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = SCALE                          # 记忆库容量
N_ACTIONS = 300                 # sever数
N_STATES = N_ACTIONS * 2 + 2       # 状态向量长度， 太接近
EPISODE = 300 -100
total_timesteps = EPISODE * SCALE

#https://zhuanlan.zhihu.com/p/260703124

"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
定义网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中。
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
"""


# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 20)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(20, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值


# 定义DQN类 (定义两个网络)
class DeepQNetwork(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        # self.model_path = sys.path[0] + '/model/' + 'bestmodel.ckpt'
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        # self.eval_net.load_state_dict(torch.load(self.model_path)['ieval net'])
        # self.target_net.load_state_dict(torch.load(self.model_path)['target net'])

        # self.model_path = 'C:/Users/Administrator/Desktop/SRL/codes/Network-Pytorch/run/bestmodel.ckpt'
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()
        # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

        # 探索新机制
        self.start_e = 1
        self.end_e = 0.05
        self.exploration_fraction = 0.8
        self.total_timesteps = total_timesteps
        self.OBSERVE = self.total_timesteps / 50

        self.loss = []
        self.q_target = []

    # def linear_schedule(self, start_e, end_e, duration, t):
    #     slope = (end_e - start_e) / duration
    #     return max(slope * t + start_e, end_e)

    def linear_schedule(self, start_e, end_e, duration, t):
        if t > self.OBSERVE:
            epsilon = start_e - (t - self.OBSERVE) * (start_e - end_e) / (self.total_timesteps - self.OBSERVE)
        elif t <= self.OBSERVE:
            epsilon = start_e
        else:
            epsilon = 0

        return epsilon
        # slope = (end_e - start_e) / duration
        # return max(slope * t + start_e, end_e)

    def choose_action(self, x, step):                                           # 定义动作选择函数 (x为状态)
        x = torch.squeeze(torch.FloatTensor(x))                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        # x = torch.from_numpy(x)
        action = []
        # interval = self.exploration_fraction * self.total_timesteps
        # EPSILON = self.linear_schedule(self.start_e, self.end_e, interval, step)
        EPSILON = self.linear_schedule(self.start_e, self.end_e, 0, step)
        # print(EPSILON)
        if np.random.uniform() > EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            # action_normal = torch.max(actions_value, 0)
            action_normal = torch.max(actions_value, 0)
            action_normal = action_normal[1].data.numpy()
            # print('*************selected action:', action_normal)
        else:                                                                   # 随机选择动作
            action_normal = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
            # print('--------------random action:',action_normal)
        return action_normal                                                          # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [[a, r]], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self, episode):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        # b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 2].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        #group loss
        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值

        # cluster loss
        loss = self.loss_func(q_eval, q_target)
        # self.loss.append(loss.data.data.numpy())
        list_qtarget = q_target.data.data.numpy().reshape(BATCH_SIZE, 1)
        for i in range(len(list_qtarget)):
            tmp = list_qtarget[i]
            self.q_target.append(tmp[0])
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数

        # https://zhuanlan.zhihu.com/p/38056115
        # if episode == 399:
        # torch.save(self.eval_net.state_dict(), self.model_path)
        # torch.save(self.target_net.state_dict(), self.model_path)
        torch.save({'epoch':episode,
                    'ieval net':self.eval_net.state_dict(),
                    "target net": self.target_net.state_dict(),
                    'optimizer':self.optimizer.state_dict()},
                     self.model_path)

