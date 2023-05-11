'''
All agents should inherit from the Agent class.
'''
import numpy as np
import sys
from .. import Agent
import cvxpy as cp


class bayes_selector_tracesAgent(Agent):
    """The bayes selector algorithm, at every iteration, solves an optimization problem for the optimal actions based on the current inventory levels and the expected number of future arrival types.  In particular, given the current state s_t denoting the available resource for the k different resource types, we solve the following optimization problem:
        :math:`\max \sum_n f_n x_n` 

        :math:`\\\t{ s. t. } 0 \leq x \leq \mathbb{E}[N_{t}]`
    where :math:`\mathbb{E}[N_{t}]` is a vector of length n with each element corresponding to the expected number of future arrivals of each type j.

    Attributes:
        epLen: The integer for episode length.
        round_flag: A boolean value that, when true, uses rounding for the action.
        config: The dictionary of values used to set up the environment.

    """

    def __init__(self, epLen, round_flag=True, dataset = None, prior = 1):
        '''Initializes the agent with attributes epLen and round_flag.

        Args:
            epLen: The integer for episode length.
            round_flag: A boolean value that, when true, uses rounding for the action.
            prior: prior variable used for the empirical distribution to avoid issues around no-data
        '''
        self.epLen = epLen
        self.round_flag = round_flag
        self.dataset = dataset
        self.prior = prior
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file

        Args:
           config: The dictionary of values used to set up the environment. '''
        self.config = config
        self.empirical_distr = self.prior * np.ones(self.config['P'].shape)
        # starting the empirical distribution, size is the [T] \times [Num Customer Types]
        # since the distribution can be time variable


        if self.dataset is not None:
            ...
            # TODO: Loop through the dataset, and add the samples to the appropriate value in the empirical distribution estimate
        return

    def reset(self):
        if hasattr(self, 'empirical_distr'): # TODO: checks if we have an empirical distribution saved and sets it back to original
            ...
        return

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs.'''

        # TODO: Note that info['customer'] contains the customer type.  Need to verify it was not a no arrival,
        # and if it wasn't, update the empirical distribution
        return

    def pick_action(self, obs, timestep):
        '''Select an action based upon the observation.

        Args:
            obs: The current state.
            timestep: The number of timesteps that have passed.
        Returns:
            list:
            action: The action the agent will take in the next timestep.'''
        # TODO: Should be the same as the other agent