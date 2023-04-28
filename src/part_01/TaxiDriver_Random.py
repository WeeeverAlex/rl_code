import gymnasium as gym
env = gym.make("Taxi-v3", render_mode='human').env

done = False
episodios = 1000

for i in range(episodios):
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        #
        # state, action, reward, next_state p/ aprender uma policy
        #
        
        #state.render()