import gym
# from utils import *
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import clear_output

# env = gym.make('FrozenLake8x8-v1')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()


env = gym.make('FrozenLake8x8-v1')
env = env.unwrapped
desc = env.unwrapped.desc

time_array=[0]*10
gamma_arr=[0]*10
iters=[0]*10
list_scores=[0]*10


def convergence_plotter(iteration, delta):
    plt.plot(iteration, delta)
    plt.xlabel('iterations')
    plt.title('Convergence')
    plt.ylabel('Utility difference')
    # plt.xaxis('scaled')
    plt.grid()
    plt.show()

# reference from https://zoo.cs.yale.edu/classes/cs470/materials/hws/hw7/FrozenLake.html and modified to my needs
def value_iteration(env, max_iterations=1000, lmbda=0.9, ns=64, na=4, e=.001):
  stateValue = [0 for i in range(ns)]
  newStateValue = stateValue.copy()
  delta = []
  iterations = []
  j=0
  for i in range(max_iterations):
    for state in range(ns):
      action_values = []      
      for action in range(na):
        state_value = 0
        for i in range(len(env.P[state][action])):
          prob, next_state, reward, done = env.P[state][action][i]
          state_action_value = prob * (reward + lmbda*stateValue[next_state])
          state_value += state_action_value
        action_values.append(state_value)      #the value of each action
        best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
        newStateValue[state] = action_values[best_action]  #update the value of the state
    delta.append(abs(sum(stateValue) - sum(newStateValue)))
    j=j+1
    iterations.append(j)
    if i > 1000: 
      if sum(stateValue) - sum(newStateValue) < e:   # if there is negligible difference break the loop
        break

    else:
      stateValue = newStateValue.copy()
  return stateValue, iterations, delta 





def get_policy(env,stateValue, lmbda=0.9, ns=64, na = 4):
  policy = [0 for i in range(ns)]
  for state in range(ns):
    action_values = []
    for action in range(na):
      action_value = 0
      for i in range(len(env.P[state][action])):
        prob, next_state, r, _ = env.P[state][action][i]
        action_value += prob * (r + lmbda * stateValue[next_state])
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[state] = best_action
  return policy 




def get_score(env, policy, episodes=1000):
  misses = 0
  steps_list = []
  for episode in range(episodes):
    observation = env.reset()
    steps=0
    while True:
      
      action = policy[observation]
      observation, reward, done, _ = env.step(action)
      steps+=1
      if done and reward == 1:
        # print('You have got the fucking Frisbee after {} steps'.format(steps))
        steps_list.append(steps)
        break
      elif done and reward == 0:
        # print("You fell in a hole!")
        misses += 1
        break
  print('----------------------------------------------')
  print('You took an average of {:.0f} steps to go across lake'.format(np.mean(steps_list)))
  print('And you fell in the hole {:.2f} % of the times'.format((misses/episodes) * 100))
  print('----------------------------------------------')


def q_learning(env, alpha = 0.1, gamma = 0.6, epsilon = 0.1):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")
    return q_table

def eval_q_learning(q_table):
    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")

print('VALUE ITERATION WITH FROZEN LAKE')
# q_table = q_learning(env)
# eval_q_learning(q_table)
statevalue, iteration, delta = value_iteration(env)
convergence_plotter(iteration, delta)
policy = get_policy(env,statevalue)
get_score(env, policy)




