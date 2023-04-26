import gymnasium as gym
import numpy as np
from numpy import loadtxt

env = gym.make('FrozenLake-v1', render_mode='ansi',map_name="8x8", is_slippery=True).env
q_table = loadtxt('data/q-table-frozen-lake-sarsa.csv', delimiter=',')

rewards = 0

for i in range(0,100):    
    (state, _) = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _, info = env.step(action)
    rewards += reward
print(rewards)