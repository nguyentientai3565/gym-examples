import gym
from gym import spaces
import pygame
import numpy as np
import random

class TrafficState:
    def __init__(self, state, load):
        self.state = state
        self.value = load

traffic_info = TrafficState(state=np.array([1, 2, 3]), load=np.array([11, 12, 13]))

traffic_state = np.empty((5, 5), dtype=TrafficState)

random_number = round(random.uniform(0, 1), 2)
#print(random_number)
#print(traffic_info.state[1])

dict = {
    0: np.array([0, 0,0, 0]),
    1: np.array([0, 0, 0, 1]),
    2: np.array([0, 0, 1, 0]),
    3: np.array([0, 0, 1, 1]),
    4: np.array([0, 1, 0, 0]),
    5: np.array([0, 1, 0, 1]),
    6: np.array([0, 1, 1, 0]),
    7: np.array([0, 1, 1, 1]),
    8: np.array([1, 0, 0, 0]),
    9: np.array([1, 0, 0, 1]),
    10: np.array([1, 0, 1, 0]),
    11: np.array([1, 0, 1, 1]),
    12: np.array([1, 1, 0, 0]),
    13: np.array([1, 1, 1, 0]),
    15: np.array([1, 1, 1, 1]),
    16: np.array([0, 0, 0, 0])

}

print(dict[15])