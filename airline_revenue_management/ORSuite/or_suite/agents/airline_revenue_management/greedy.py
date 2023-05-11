'''
All agents should inherit from the Agent class.
'''
import numpy as np
import sys
from .. import Agent
import cvxpy as cp


class greedyAgent(Agent):
    """The greedy agent, at every iteration, accepts any requests if feasible.

    Attributes:
        epLen: The integer for episode length.
        config: The dictionary of values used to set up the environment.
    """

    def __init__(self, epLen):
        '''Initializes the agent with attributes epLen and round_flag.

        Args:
            epLen: The integer for episode length.
            round_flag: A boolean value that, when true, uses rounding for the action.
        '''
        self.epLen = epLen
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file

        Args:
           config: The dictionary of values used to set up the environment. '''
        self.config = config
        return

    def reset(self):
        pass

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs.'''
        return

    def pick_action(self, obs, timestep):
        '''Select an action based upon the observation.

        Args:
            obs: The current state.
            timestep: The number of timesteps that have passed.
        Returns:
            list:
            action: The action the agent will take in the next timestep.'''
        # use the config to populate vector of the demands
        num_type = len(self.config['f'])
        action = np.asarray([1 for i in range(num_type)])
        return action
