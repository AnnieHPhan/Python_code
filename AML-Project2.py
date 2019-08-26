import numpy as np
import random
import pyglet
import gc
import time

from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, TensorBoard
import keras.backend as K
import json
from mini_pacman import PacmanGame

with open('test_params.json', 'r') as file:
    read_params = json.load(file)
game_params = read_params['params']
env = PacmanGame(**game_params)

env.render()

from tabulate import tabulate

def create_dqn_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Dense(units=1000, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

action_to_dxdy = {1: (1, -1),
                  2: (1, 0),
                  3: (1, 1),
                  4: (0, -1),
                  5: (0, 0),
                  6: (0, 1),
                  7: (-1, -1),
                  8: (-1, 0),
                  9: (-1, 1)}

def get_state(obs):
    v = []
    x,y = obs['player']
    v.append(x)
    v.append(y)
    for x, y in obs['monsters']:
        v.append(x)
        v.append(y)
    for x, y in obs['diamonds']:
        v.append(x)
        v.append(y)
    for x, y in obs['walls']:
        v.append(x)
        v.append(y)
    return v

class QLearn:
    def __init__(self, gamma=0.95, alpha=0.05):
        from collections import defaultdict
        self.gamma = gamma
        self.alpha = alpha
        self.qmap = defaultdict(int)

    def iteration(self, old_state, old_action, reward, new_state, new_possible_actions):
        # Produce iteration step (update Q-Value estimates)
        old_stateaction = tuple(old_state) + (old_action,)
        max_q = max([self.qmap[tuple(new_state) + (a,)] for a in new_possible_actions])
        self.qmap[old_stateaction] = (1-self.alpha)*self.qmap[old_stateaction] + self.alpha*(reward+self.gamma*max_q)
        return

    def best_action(self, state, possible_actions):
        # Get the action with highest Q-Value estimate for specific state
        a, q = max([(a, self.qmap[tuple(state) + (a,)]) for a in possible_actions], key=lambda x: x[1])
        return a
input_shape = (len(get_state(env.reset())),)
nb_actions = len(action_to_dxdy)
online_network = create_dqn_model(input_shape, nb_actions)
online_network.compile(optimizer=Adam(), loss='mse')
target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(online_network).create(prog='dot', format='svg'))
from keras.utils import plot_model
plot_model(online_network, to_file='online_network.png',show_shapes=True,show_layer_names=True)

from collections import deque
replay_memory_maxlen = 1_000_000
replay_memory = deque([], maxlen=replay_memory_maxlen)

def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(n_outputs)  # random action
    else:
        return np.argmax(q_values)  # q-optimal action

n_steps = 100_000 # number of times 
warmup = 1_000 # first iterations after random initiation before training starts
training_interval = 4 # number of steps after which dqn is retrained
copy_steps = 2_000 # number of steps after which weights of 
                   # online network copied into target network
gamma = 0.99 # discount rate
batch_size = 64 # size of batch from replay memory 
eps_max = 1.0 # parameters of decaying sequence of eps
eps_min = 0.05
eps_decay_steps = 50_000
obs = env.reset()
q_values = online_network.predict(np.array([get_state(obs)]))
epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
action = epsilon_greedy(q_values, epsilon, nb_actions)

step = 0
iteration = 0
done = True
warmup = 64

while step < n_steps:
    if done:
        obs = env.reset()
    iteration += 1
    q_values = online_network.predict(np.array([get_state(obs)]))  
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action = epsilon_greedy(q_values, epsilon, nb_actions)
    next_obs= env.make_action(action + 1)
    reward = next_obs['reward']
    done = next_obs['end_game']
    replay_memory.append((obs, action, reward, next_obs, done))
    obs = next_obs

    if iteration >= warmup and iteration % training_interval == 0:
        step += 1
        minibatch = random.sample(replay_memory, batch_size)
        replay_state = np.array([get_state(x[0]) for x in minibatch])
        replay_action = np.array([x[1]for x in minibatch])
        replay_rewards = np.array([x[2] for x in minibatch])
        replay_next_state = np.array([get_state(x[3]) for x in minibatch])
        replay_done = np.array([x[4] for x in minibatch], dtype=int)
        target_predict = target_network.predict(replay_next_state)
        target_for_action = replay_rewards + (1-replay_done) * gamma * \
                                    np.amax(target_network.predict(replay_next_state), axis=1)
        target = online_network.predict(replay_state)
        target[np.arange(batch_size), replay_action] = target_for_action  
        online_network.fit(replay_state, target, epochs=step, verbose=1, initial_epoch=step-1)
        if step % copy_steps == 0:
            target_network.set_weights(online_network.get_weights())


from keras.models import load_model


def test_dqn_strategy(obs):
    q_values = online_network.predict(np.array([get_state(obs)]))
    action = epsilon_greedy(q_values,0.05,obs['possible_actions'])
    return action+1

from mini_pacman import test
test(strategy=test_dqn_strategy, log_file='test_pacman_log.json')
