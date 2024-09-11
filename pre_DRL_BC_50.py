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
environment = environment.environment(20)
# DQN = DeepQNetwork()
# DQN.model_path = sys.path[0] + '/model/TS/' + 'bestmodel.ckpt'

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

    def Asynchronous_BC_learn(task, machines, agent):
        u = [0.6, 0.4]
        n_samples = 30
        GA_states, GA_actions, GA_reward = GA_Expert_data(task, machines)
        PSO_states, PSO_actions, PSO_reward = PSO_Expert_data(task, machines)

        sample_indices = np.random.randint(low=0,
                                           high=GA_states.shape[0],
                                           size=n_samples)
        if GA_reward > PSO_reward:
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[0])  
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[1]) 
        else:
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[1])  
            agent.BC_learn(GA_states[sample_indices], GA_actions[sample_indices], u[0]) 
        # print('-----------threading end-----------')

    environment.reset()
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

    done = False
    state_list = []
    action_list = []
    next_state_list = []
    done_list = []
    # return_list = []

    timer_t = RepeatTimer(0.2, running_machine)
    timer_t.start()
    timer_p = RepeatTimer(0.5, power_machine)
    timer_p.start()

    while not done:
        # print('step:', step)
        state = environment.observe(environment.current_task)
        action = agent.take_action(state)
        # print('action:', action)
        ooob = copy.deepcopy(state)
        next_state, reward, done = environment.step(ooob, action, False) 

        replay_buffer.add(state[0], action, reward, next_state[0], done)

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

        if (step % 30 == 0):
            # info = copy.deepcopy(environment)
            current_task_num = environment.task_allocate_counter
            if current_task_num + 29 <= environment.task_number:
                tasks = environment.tasks[current_task_num:(current_task_num + 29)]
            else:
                tasks = environment.tasks[current_task_num:]
            machines = copy.deepcopy(environment.machines)
            t = Thread(target=Asynchronous_BC_learn(tasks, machines, agent), name=str(step) + "emulator")
            t.start() 

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
