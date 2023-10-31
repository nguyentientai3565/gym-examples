import gym
from gym import spaces
import pygame
import numpy as np
import random
import logging
from datetime import datetime

LOGGING_FILE = f"logs/app-{datetime.today().strftime('%Y-%m-%d')}.log"
class TrafficState:
    def __init__(self, state, load):
        self.state = state
        self.load = load

class NetworkEnv(gym.Env):
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    metadata = None

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        #self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with 2 matrics
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "Demand": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "State": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 17 actions, corresponding to 0000->1111
        self.action_space = spaces.Discrete(17)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
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
            13: np.array([1, 1, 0, 1]),
            14: np.array([1, 1, 1, 0]),
            15: np.array([1, 1, 1, 1]),
            16: np.array([0, 0, 0, 0])

        }

        assert render_mode is None or render_mode in self.metadata
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        #self.window = None
        #self.clock = None

    def _get_obs(self):
        state = np.zeros((5,5))
        load = np.zeros((5,5))
        for i in range(self.size):
            for j in range(self.size):
                load[i][j] = self._traffic_state[i][j].load
                state[i][j] = self._traffic_state[i][j].state
        return {"traffic_state(state)": state,
                 "traffic_state(load)": load
                }


# return loss or traffic_converage and energy_saving  ( chuua biet cach tinh energy_saving vi kb P la gi?)
    def _get_info(self):
        total = np.zeros((5,5))
        for i in range(self.size):
            for j in range(self.size):
                total[i][j] = self._traffic_state[i][j].load
        return {
            "traffic_loss": np.sum(self._traffic_demand) - np.sum(total),
            "traffic_coverage": np.sum(total)/np.sum(self._traffic_demand)
        }


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the TRAFFIC_DEMAND dij = 0.1
        self._traffic_demand = np.ones((5, 5)) * 0.1

        # tao ra ma tran traffic_state random
        self._traffic_state = np.empty((self.size, self.size), dtype=TrafficState)
        for i in range(self.size):
            for j in range(self.size):
                #state=random.randint(0,1)
                #self._traffic_state[i][j] = TrafficState(state=state, load = 0 if state == 0 else round(random.uniform(0, 1), 2))
                self._traffic_state[i][j] = TrafficState(state= 1, load = 0.1)

        #self._traffic_state = self._traffic_state
        #while np.array_equal(self._traffic_demand, self._traffic_state):
            #self._traffic_state = self.np_random.integers(
                #0, self.size, size=2, dtype=int
            #)
            #self._traffic_state

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self):
        action = 0
        neighbors = 0
        # random toi 1 BS bat ky => BS[i,j] 
        i = random.randint(0, self.size - 1)
        j = random.randint(0, self.size - 1)
        #i=0
        #j =4
        #print(i, j)
        
        # xet BS[i, j] neu = 0, no dang tat thi bat no len va giu tai cua no = 0
        if(self._traffic_state[i][j].load == 0 and self._traffic_state[i][j].state == 0):
            self._traffic_state[i][j].state = 1
        # If load = 0 and state = 1 => do nothing
        elif(self._traffic_state[i][j].load == 0 and self._traffic_state[i][j].state == 1):
            action = 16
            print("Do nothing")
        # if load != 0 and ON => OFF and share load 
        else:
            # xet cac nut lan can
            #TH1 xet cac nut o giua, tiep xuc 4 huong va kiem tra cac nut ON/OFF => action
            if( i - 1 >= 0 and j- 1 >= 0 and i + 1 < self.size and j + 1 < self.size):
                neighbors = 4
                if(self._traffic_state[i][j - 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 2)
                if(self._traffic_state[i - 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 3)
                if(self._traffic_state[i][j + 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 4)
                if(self._traffic_state[i + 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 5)
            
            #Th2 xet cac nut co 1 bien
            if( i - 1 < 0 and j- 1 >= 0 and j  + 1 < self.size):
                neighbors = 3
                if(self._traffic_state[i][j - 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 2)
                if(self._traffic_state[i][j + 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 4)
                if(self._traffic_state[i + 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 5)

            #th3
            if( i + 1 >= self.size and j- 1 >= 0 and j  + 1 < self.size):
                neighbors = 3
                if(self._traffic_state[i][j - 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 2)
                if(self._traffic_state[i][j + 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 4)
                if(self._traffic_state[i - 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 3)
            #th4
            if( j - 1 < 0 and i - 1 >= 0 and i + 1 < self.size):
                neighbors = 3
                if(self._traffic_state[i - 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 3)

                if(self._traffic_state[i][j + 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 4)
                if(self._traffic_state[i + 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 5)
            #th5
            if( j + 1 >= self.size and i - 1 >= 0 and i + 1 < self.size):
                neighbors = 3
                if(self._traffic_state[i - 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 3)

                if(self._traffic_state[i][j -1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 2)
                if(self._traffic_state[i + 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 5)
            #th6
            if((i - 1 < 0 and j- 1 < 0)):
                neighbors = 2
                if(self._traffic_state[i][j + 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 4)
                if(self._traffic_state[i + 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 5)
            #th7
            if((i + 1 >= self.size and j + 1 >= self.size)):
                neighbors = 2
                if(self._traffic_state[i][j - 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 2)
                if(self._traffic_state[i - 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 3)
            #th8
            if((i - 1 < 0 and j + 1 >= self.size)):
                neighbors = 2
                if(self._traffic_state[i + 1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 5)
                if(self._traffic_state[i][j - 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size -2)

            #th9
            if((j - 1 < 0 and i + 1 >= self.size)):
                neighbors = 2
                if(self._traffic_state[i][j + 1].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 4)
                if(self._traffic_state[i -1][j].state == 0 ):
                    neighbors = neighbors -1
                else: 
                    action = action + 2**(self.size - 3)

        #share load and OFF BS
        if(neighbors!=0):
            load_share = self._traffic_state[i][j].load / neighbors
        self._traffic_state[i][j].load = 0
        self._traffic_state[i][j].state = 0

        #
        #print(action)
        direction = self._action_to_direction[action]

        for index, act in enumerate(direction):
            if(index == 0 and act == 1):
                self._traffic_state[i][j -1].load += load_share
                if(self._traffic_state[i][j -1].load > 1):
                    self._traffic_state[i][j -1].load = 1
            if(index == 1 and act == 1):
                self._traffic_state[i -1][j].load += load_share
                if(self._traffic_state[i -1][j].load > 1):
                    self._traffic_state[i-1][j].load = 1
            if(index == 2 and act == 1):
                self._traffic_state[i ][j + 1].load += load_share
                if(self._traffic_state[i ][j + 1].load > 1):
                    self._traffic_state[i][j + 1].load = 1
            if(index == 3 and act == 1):
                self._traffic_state[i +1][j].load += load_share
                if(self._traffic_state[i +1][j].load > 1):
                    self._traffic_state[i +1][j].load = 1
        # We use `np.clip` to make sure we don't leave the grid
        
        # An episode is done iff the agent has reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location)
        #reward = 1 if terminated else 0  # Binary sparse rewards
        terminated = 1
        Po = 130 #W
        n = 4.7 
        Pt = 20 #W
        Ps = 75 #w
        total_Pmax = 25 * (Po + 0.1 *n * Pt) 
        P_matrix = np.zeros((5,5))
        if(terminated):
            for i in range(self.size):
                for j in range(self.size):
                    if(self._traffic_state[i][j].state != 0):
                        P_matrix[i][j] = Po + n * self._traffic_state[i][j].load * Pt
                    else:
                        P_matrix[i][j] = Ps
                    #print(P_matrix[i][j])
            reward = total_Pmax - np.sum(P_matrix) - 100*self._get_info()["traffic_loss"]
         
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info




if __name__ == "__main__":
    object1 = NetworkEnv()
    print(object1.action_space)
    print(object1.observation_space)
    logging.basicConfig(filename=LOGGING_FILE, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    object1.reset()
    #print(object1._get_obs())
    #print(object1._get_info())

    #object1.step()
    #observation, reward, terminated, truncated, info = object1.step()
    #print(reward)
    #print(object1._get_obs())
    #print(object1._get_info())
    for i in range(7):
        logging.info(f"Day : {i+1}")
        for _ in range(24):
            observation, reward, terminated, truncated, info = object1.step()
            logging.info(f"reward: {reward}")
            logging.info(f"info: {info}")

