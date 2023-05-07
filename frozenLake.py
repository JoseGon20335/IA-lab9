import gym
import random
import numpy as np
from collections import defaultdict

env = gym.make('FrozenLake-v1', is_slippery=True)

alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate


def q_learning(env, num_episodes):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i in range(num_episodes):
        state = env.reset()
        state = tuple(state.items())  # convert dictionary to tuple
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # exploration
            else:
                action = np.argmax(Q[state])  # exploitation

            next_state, reward, done, _ = env.step(action)
            # convert dictionary to tuple
            next_state = tuple(next_state.items())
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state

    return Q


Q = q_learning(env, 10000)

state = env.reset()
done = False
while not done:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    env.render()
