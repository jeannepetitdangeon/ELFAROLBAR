import os
os.chdir(r"C:\Users\samue\Documents\GitHub\ELFAROLBAR")

import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from DDPG import *
# from Trainer import *
# from models import * 

env = NormalizedEnv()


agent = DDPGagent()
#noise = OUNoise(env.action_space)


batch_size = 128
rewards = []
actions=[]
avg_rewards = []
epsilon = 0.7
for episode in range(0,2000):
    state = np.array([0])
    #state = env.reset()
    #noise.reset()
    episode_reward = 0
    
    for step in range(500):
        
        action = agent.get_action(state=state,exploration_rate=epsilon)      
        print()
        new_state, reward, done= env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        print("Len Memory", len(agent.memory))
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {}, action:{} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:]),action))
            break

    rewards.append(episode_reward)
    actions.append(action)
    avg_rewards.append(np.mean(rewards[-10:]))
    epsilon = epsilon-0.01*episode

print(f"Number of event nights: {env.get_event_night_count()} and Number of non-event nights: {env.get_non_event_night_count()}")

#average_event_actions = np.mean(event_actions)
#average_non_event_actions = np.mean(non_event_actions)

#print(f"Average action for event nights: {average_event_actions}")

non_event_actions = env.get_non_event_actions()
print(f"Number of actions for non-event nights: {len(non_event_actions)}")

if len(non_event_actions) > 0:
    average_non_event_actions = np.mean(non_event_actions)
    print(f"Average action for non-event nights: {average_non_event_actions}")
else:
    print("No actions recorded for non-event nights.")

event_actions = env.get_event_actions()
print(f"Number of actions for event nights: {len(event_actions)}")

if len(event_actions) > 0:
    average_event_actions = np.mean(event_actions)
    print(f"Average action for event nights: {average_event_actions}")
else:
    print("No actions recorded for event nights.")




event_night_count = env.get_event_night_count()
non_event_night_count = env.get_non_event_night_count()
event_actions = env.get_event_actions()
non_event_actions = env.get_non_event_actions()

# Create arrays to represent episode numbers for event and non-event nights
event_episodes = np.arange(len(event_actions))
non_event_episodes = np.arange(len(non_event_actions))

# Create scatter plots for event and non-event nights with smaller dots
plt.scatter(event_episodes, event_actions, marker='o', color='red', label='Event Nights', s=10)
plt.scatter(non_event_episodes, non_event_actions, marker='x', color='blue', label='Non-Event Nights', s=10)

# Set labels and title
plt.xlabel('Episode')
plt.ylabel('Actions')
plt.title('Scatter Plot of Actions on Event and Non-Event Nights')
plt.legend()

# Show the plot
plt.show()

#plt.plot(rewards)
#plt.plot(avg_rewards)
#plt.plot(actions)
#plt.plot()
#plt.xlabel('Episode')
#plt.ylabel('actions')
#plt.ylabel('Reward')
#plt.show()

#event_nights = [i for i in range(len(rewards)) if env.event_night_count > 0]
#non_event_nights = [i for i in range(len(rewards)) if i not in event_nights]

#event_actions = [actions[i] for i in event_nights]
#non_event_actions = [actions[i] for i in non_event_nights]

# Create box plots for event nights and non-event nights
#plt.boxplot([non_event_actions, event_actions], labels=['Non-Event Nights', 'Event Nights'])

# Set labels and title
#plt.xlabel('Night Type')
#plt.ylabel('Actions')
#plt.title('Box Plot of Actions on Event and Non-Event Nights')

# Show the plot
#plt.show()

