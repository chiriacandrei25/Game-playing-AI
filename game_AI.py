import gym
import random
import numpy as np
import  tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

learning_rate = 0.001
environment = gym.make("CartPole-v1")
environment.reset()
steps_target = 500
score_requirement = 50
initial_games = 1000

def some_random_games_first():
    for episode in range(5):
        environment.reset()
        for step in range(steps_target):
            environment.render()
            action = environment.action_space.sample()
            observation, reward, done, info = environment.step(action)
            if done:
                break

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    prev_observation = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        for _ in range(steps_target):
            action = random.randrange(0, 2)
            observation, reward, done, info = environment.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                else:
                    output = [1, 0]
                training_data.append([data[0], output])
        environment.reset()
        scores.append(score)
        training_data_saved = np.array(training_data)
        np.save("saved.npy", training_data_saved)

    # Statistics data
    print("Average accepted score:", mean(accepted_scores))
    print("Median accepted score:", median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

initial_population()

