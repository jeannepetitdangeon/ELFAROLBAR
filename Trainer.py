import os
os.chdir(r"C:\Users\samue\Documents\GitHub\ELFAROLBAR")
import numpy as np


import gym
from collections import deque
import random

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv():
    """ Wrap action """
    
    def __init__(self):
        self.event_night_count = 0
        self.non_event_night_count = 0
        self.event_actions = []  # List to store actions on event nights
        self.non_event_actions = []  # List to store actions on non-event nights
        
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    
    def reward(self,individual_actions,optimal_crowd=60):
        crowd = sum(individual_actions)
        rewards = list()
        if crowd == 0:
            rewards = [-5 for i in range(0,100)]
        elif crowd < 0.5 * optimal_crowd:
            rewards= [-5 if i ==1 else 0 for i in individual_actions]
        elif crowd < 0.9 * optimal_crowd:
            rewards = [1 if i ==1 else 0 for i in individual_actions]
        elif crowd < 1.1 * optimal_crowd:
            rewards = [15 if i ==1 else 0 for i in individual_actions]
        elif crowd < 1.3 * optimal_crowd:
            rewards = [1 if i ==1 else 0 for i in individual_actions]
        else:
            rewards = [-5 if i ==1 else 0 for i in individual_actions]    
        return sum(rewards)/100
    

#noise = np.random.normal(0, 1, len(rewards))  # Generating noise for each reward
#rewards = [reward + noise[idx] for idx, reward in enumerate(rewards)]

        
    def step(self, action):
        
        self.event_night = np.random.choice([True, False], p=[1 / 7, 6 / 7])
        if self.event_night:
            self.event_night_count += 1
        else:
            self.non_event_night_count += 1
            
        individual_actions = list()
        for i in range(0, 100):
            if self.event_night:
                individual_actions.append(1)  # Go to the bar on event night
            else:
                if random.uniform(0, 1) <= action:
                    individual_actions.append(1)  # Go to the bar on non-event night
                else:
                    individual_actions.append(0)  # Don't go to the bar on non-event night
    
        new_state = 0
        rewards_score = self.reward(individual_actions)
        if self.event_night:
            rewards_score += 2
        
        # Collect actions for both event and non-event nights
        if self.event_night:
            self.event_actions.append(action)
        else:
            self.non_event_actions.append(action)
    
        done = True
        return new_state, rewards_score, done


    def get_event_night_count(self):
        return self.event_night_count

    def get_non_event_night_count(self):
        return self.non_event_night_count

    def get_event_actions(self):
        return self.event_actions

    def get_non_event_actions(self):
        return self.non_event_actions
    
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)


    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
