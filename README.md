# ELFAROLBAR

## Summary

- [Description](#description)
- [Structure du projet](#structure)
- [Pré-requis](#pré-requis)
- [Installation](#installation)
- [Technologie](#technologie)
- [Auteurs](#auteurs)

## Description

The El Farol Bar problem is about the probability to go to a bar or not following the crowd. We suppose that the capacity of the bar is a hundred people and that the optimum crowd is 60 people. Using a reinforcement learning algorithm, we train an agent to use a good policy to have a maximum reward when going to the bar. 
We also add an extension, considering there is an event every seven days, you get another reward, in addition to the first reward function. We will also analyze these results following the kind of reward we add for the event, to see how it affects the behaviour of the agent.  

## Structure

- In [Graphs](https://github.com/jeannepetitdangeon/ELFAROLBAR/tree/main/Graphs), you will find every visual we made from the different tests we made by changing the reward function. 
- In [DDPG](https://github.com/jeannepetitdangeon/ELFAROLBAR/blob/main/DDPG.py), you will find our code defining the DDPG agent with actor-critic networks and implementing the DDPG algorithm. 
- In [Main](https://github.com/jeannepetitdangeon/ELFAROLBAR/blob/main/Main.py), you will find the main code, linking every parts of the model. 
- In [Trainer](https://github.com/jeannepetitdangeon/ELFAROLBAR/blob/main/Trainer.py), you will find the part of our code that trains the agent. 
- In [Model](https://github.com/jeannepetitdangeon/ELFAROLBAR/blob/main/Model.py), you will find the part of our code defining the neural network architectures for the Critic and the Actor.
- In [Paper](), you will find our student paper, describing our model and the extension we add as well as our results and self critics. 

## Requirements 

1. A computer using Windows or macOS
2. [Python](https://www.python.org/downloads/) ou an environment compatible with Python. 

## Installation

- Install the repository in a folder in your computer
- Start all the parts of the code then the ```main``` code on a Python kernel. 

## Technologies

- [Pettingzoo](https://pettingzoo.farama.org/index.html)
- [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

## Auteurs

- Jeanne PETIT-DANGEON : [@jeannepetitdangeon](https://github.com/jeannepetitdangeon)
- Samuel DOBROSSY : [@SamiGITHUB01](https://github.com/SamiGITHUB01)
