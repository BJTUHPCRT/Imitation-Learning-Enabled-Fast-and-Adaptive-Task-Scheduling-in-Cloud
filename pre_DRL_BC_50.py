# from comparsionAlgo.RR import RoundRobinAgent
from entities import environment
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from threading import Timer
import torch
from networks.BC import GA_Expert_data
from networks.BC import PSO_Expert_data
from networks.BC import AOILA_algo3
import random
import numpy as np
import time
from entities.rl_utils import ReplayBuffer
from networks.BC import ActorCritic
from threading import Thread

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

global observation
# 初始化环境和网络
environment = environment.environment(20)
# DQN = DeepQNetwork()
# DQN.model_path = sys.path[0] + '/model/TS/' + 'bestmodel.ckpt'

# 专家策略 RR
class RoundRobinAgent:
    def __init__(self, action_len):
        self.action_len = action_len - 1
        self.tmp_action = 0

    def take_action(self):
        action = self.tmp_action
        if self.tmp_action == self.action_len:
            self.tmp_action = 0
        else:
            self.tmp_action += 1
        return action

class RandomAgent:
    def __init__(self, action_len):
        self.action_len = action_len - 1
        self.tmp_action = 0

    def take_action(self):
        return random.randint(0, self.action_len)

RR_agent = RoundRobinAgent(environment.action_len)
Random_agent = RandomAgent(environment.action_len)

actor_lr = 0.0000025
critic_lr = 0.0000025
# actor_lr = 0.01
# critic_lr = 0.01
num_episodes = 1
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
lr = 1e-3
state_dim = environment.action_len * 2 + 2
action_dim = environment.action_len
# bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr)
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device, True, 50)
n_iterations = 300
batch_size = 64
test_returns = []
buffer_size = 10000
minimal_size = 64
replay_buffer = ReplayBuffer(buffer_size)

# # 获取专家数据
# n_episode = 5  # 只生成了一条轨迹
# expert_s, expert_a = sample_RR_data(n_episode, environment, RR_agent)
#
# n_samples = 30  # 采样30个数据
# random_index = random.sample(range(expert_s.shape[0]), n_samples)
# expert_s = expert_s[random_index] #(batch, h, w)
# expert_a = expert_a[random_index]

# with tqdm(total=n_iterations, desc="进度条") as pbar:
#     step = 0
#     def running_machine():
#         for i in range(len(environment.machines)):
#             if environment.machines[i].turnOn == True:
#                 environment.machines[i].check_if_have_task_finished()
#     def power_machine():
#         global timer_p
#         environment.compute_power()
#         environment.compute_latency()
#
#     for i in range(n_iterations): # 重复利用专家经验训练网络
#         sample_indices = np.random.randint(low=0,
#                                            high=expert_s.shape[0],
#                                            size=batch_size)
#         bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices]) #用专家经验 用BC算法训练网络
#         # 可以用自己的经验，训练BC网络，完成自己的学习，一段时间进行一次专家经验的学习
#         # 或者将专家经验混合自己的经验混合，然后再学习，防止震荡
#
#         # 测试agent的表现() test_agent
#         return_list = []
#         n_episode = 5
#         for episode in range(n_episode):
#             # 开启子线程
#             timer_t = RepeatTimer(0.2, running_machine)
#             timer_t.start()
#
#             timer_p = RepeatTimer(0.5, power_machine)
#             timer_p.start()
#
#             episode_return = 0
#             environment.reset()
#             state = environment.observe(environment.current_task)
#             done = False
#             while not done:
#                 action = bc_agent.take_action(state)
#                 ooob = copy.deepcopy(state)
#                 next_state, reward, done = environment.step(ooob, action)
#                 state = next_state
#                 episode_return += reward
#             return_list.append(episode_return)
#
#         current_return = np.mean(return_list)
#         test_returns.append(current_return)
#         if (i + 1) % 10 == 0:
#             pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
#         pbar.update(1)

# off policy 更新方式学习
return_list = []
lstep = 0

for i_episode in range(1):
    step = 0
    def running_machine():
        for i in range(len(environment.machines)):
            if environment.machines[i].turnOn == True:
                environment.machines[i].check_if_have_task_finished()

    def power_machine():
        global timer_p
        environment.compute_power()
        environment.compute_latency()

    # 线程执行完当前函数后会自动退出，不需要人为终止
    def Asynchronous_BC_learn(task, machines, agent):
        u = [0.6, 0.4]
        n_samples = 30
        # 使用专家1调度30个任务,获取轨迹
        GA_states, GA_actions, GA_reward = GA_Expert_data(task, machines)
        # 使用专家2调度30个任务,获取轨迹
        PSO_states, PSO_actions, PSO_reward = PSO_Expert_data(task, machines)

        # 因为是取用所有states,所以不需要有batch概念
        sample_indices = np.random.randint(low=0,
                                           high=GA_states.shape[0],
                                           size=n_samples)
        # 使用专家知识更新网络
        if GA_reward > PSO_reward:
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[0])  # 用专家经验 用BC算法训练网络
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[1])  # 用专家经验 用BC算法训练网络
        else:
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[1])  # 用专家经验 用BC算法训练网络
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[0])  # 用专家经验 用BC算法训练网络
        # print('-----------threading end-----------')

    environment.reset()
    # 每次都存储一个episode的序列
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

    done = False
    state_list = []
    action_list = []
    next_state_list = []
    done_list = []
    # return_list = []

    # 开启子线程
    # 可能pytorch的运行太慢了，需要调大值
    timer_t = RepeatTimer(0.2, running_machine)
    timer_t.start()
    timer_p = RepeatTimer(0.5, power_machine)
    timer_p.start()

    while not done:
        # print('step:', step)
        state = environment.observe(environment.current_task)
        action = agent.take_action(state)  # 不需要和环境互动，直接用当前策略产生轨迹
        # print('action:', action)
        ooob = copy.deepcopy(state)
        next_state, reward, done = environment.step(ooob, action, False)  # 此时需要从环境获取下一个state

        replay_buffer.add(state[0], action, reward, next_state[0], done)

        # 当buffer数据的数量超过一定值后,才进行Q网络训练
        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            agent.update(transition_dict)
            lstep += 1
            # print('learning step:', lstep)

        # BC 进行知识补充
        # 应该重新开一个线程，不能影响主程序的任务分配
        # 每30个任务开启一个仿真器,进行一次专家指导
        if (step % 30 == 0):
            # 为了让程序更快一点
            # info = copy.deepcopy(environment)
            current_task_num = environment.task_allocate_counter
            if current_task_num + 29 <= environment.task_number:
                tasks = environment.tasks[current_task_num:(current_task_num + 29)]
            else:
                tasks = environment.tasks[current_task_num:]
            machines = copy.deepcopy(environment.machines)
            t = Thread(target=Asynchronous_BC_learn(tasks, machines, agent), name=str(step) + "emulator")  # 线程对象
            t.start()  # 启动线程

        # break while loop when end of this episode
        if done:
            print('***********************************{}***************************************'.format(i_episode))
            while True:
                if environment.compute_executed_taskNumber() == False:
                    time.sleep(2)
                else:
                    timer_p.cancel()
                    timer_t.cancel()
                    break
            # print('execute_taskNumber: ', environment.compute_executed_taskNumber())
            print('power counter', len(environment.total_power))
            environment.compute_average_response_time()
            break
        step += 1
        print('step:', step)

    print('reward:', sum(environment.reward),
          'response_time:', environment.average_response_time,
          'energy:', environment.energy)

# 保存网络参数
# torch.save({
#             # 'epoch': episode,
#             'ieval net': agent.actor.state_dict(),
#             "target net": agent.critic.state_dict(),
#             'actor optimizer': agent.actor_optimizer.state_dict(),
#             'critic optimizer': agent.critic_optimizer.state_dict()},
#             agent.model_path)


import pandas
import sys
path = sys.path[0] + '/results/ILEOFA/'
plt.clf()
episodes_list = list(range(len(environment.reward)))
dataframe = pandas.DataFrame(data=list(environment.reward), index=None, columns=None)
dataframe.to_csv(path + 'ILEOFA_50_pre_reward.csv', index=False, sep=',')
plt.plot(episodes_list, environment.reward)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('Actor-Critic on {}'.format(''))
plt.show()
