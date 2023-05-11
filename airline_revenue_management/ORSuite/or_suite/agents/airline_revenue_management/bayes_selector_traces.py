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

        if self.dataset is not None:
            for (timestep, cust) in self.dataset:
                    if cust < self.config['A'].shape[1]: # Observed customer was an actual customer and not a "no arrival"
                        self.empirical_distr[timestep, cust] += 1
        return

    def reset(self):
        if hasattr(self, 'empirical_distr'):
            self.empirical_distr = self.prior * np.ones(self.config['P'].shape)
        return

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs.'''
        if info['customer'] < self.config['A'].shape[1]: # Observed customer was an actual customer and not a "no arrival"
            self.empirical_distr[timestep, info['customer']] += 1
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

        future_distr = self.empirical_distr[timestep:, :]
        normalize_distr = future_distr / future_distr.sum(axis=0, keepdims=1)
        # print(normalize_distr)
        expect_type = np.sum(normalize_distr, axis=0)
        # gets the expected number of customer arrivals


        x = cp.Variable(num_type)
        objective = cp.Maximize(self.config['f'].T @ x)
        constraints = []
        constraints += [0 <= x]
        constraints += [x <= expect_type]

        constraints += [self.config['A'] @ x <= obs]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        # enforcing rounding rule here, add a trigger to do the other version somehow as well
        if self.round_flag:
            action = np.asarray([1 if x.value[i] / expect_type[i] >= 1/2 and np.all(np.less_equal(
                np.transpose(self.config['A'])[i], obs)) else 0 for i in range(num_type)])
        else:
            action = np.asarray([1 if np.random.binomial(1, np.minimum(1, np.maximum(
                0, x.value[i] / expect_type[i])), size=None) == 1 and np.all(np.less_equal(
                    np.transpose(self.config['A'])[i], obs)) else 0 for i in range(num_type)])
        return action
