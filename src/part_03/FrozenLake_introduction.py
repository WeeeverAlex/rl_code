import gymnasium as gym
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True).env

#
# is_slippery=False torna o ambiente deterministico
#

print(env.observation_space.n)
print(env.action_space.n)

# inicializando o ambiente
state = env.reset()

# imprimindo o ambiente
print(env.render())

# eh possivel fazer env.render(mode='ansi') para imprimir no terminal 

print('Executando algumas acoes')

print('\n\n')

print('indo para baixo')
state, reward, done, truncated, info = env.step(1)
print(env.render())
print(reward, done)

print('indo para baixo')
state, reward, done, _, _ = env.step(1)
print(env.render())
print(reward, done)

print('indo para direita')
state, reward, done, _, info = env.step(2)
print(env.render())
print(reward, done)

print('indo para direita')
state, reward, done, _, info = env.step(2)
print(env.render())
print(reward, done)

print('indo para baixo')
state, reward, done, _, info = env.step(1)
print(env.render())
print(reward, done)

print('indo para direita')
state, reward, done, _, info = env.step(2)
print(env.render())
print(reward, done)
