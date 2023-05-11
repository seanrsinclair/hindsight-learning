
import os
import timeit
from typing import List
import time
import math
import numpy as np
from numpy import core
from pulp import PULP_CBC_CMD, GLPK, GUROBI_CMD, LpInteger, LpMaximize, LpProblem, LpStatus, LpVariable, lpSum, LpContinuous, LpMinimize


DEBUG = False


"""
    Class used for the generating the IP model for the look-ahead bin packing model.
"""

class BinPackingILP():
    def __init__(self, env): # saves the environment and total capacity of bins
        self.env = env
        self.total_capacity = env.bin_capacity
    
    def _get_initial_bin(self, bin_vals): # used to get number of bins required and their (potential) initial capacity

        if bin_vals is not None:
            self._bin_num = len(self._future_item_list) + np.sum(bin_vals)
            self.current_capacity = sum([[i+1 for _ in range(bin_vals[i])] for i in range(len(bin_vals))], [])
            # converts the state representation to a list of the number of bins and their current utilization levels
            self.current_capacity = np.append(self.current_capacity, np.zeros(self._bin_num - len(self.current_capacity)))
            # maximum new additional bins is again number of items arriving
        else:
            # no initial capacity passed, so max number of bins is length of items arriving and current capacity is just zeros
            self._bin_num = len(self._future_item_list)
            self.current_capacity = np.zeros(self._bin_num)

    def _init_variables(self):

        def _init_with_shape(shape: tuple):
            return np.zeros(shape, dtype=np.int16).tolist()

        # Initialize the Used-Bin mapping variable
        self._bin_indicator = _init_with_shape(shape=(self._bin_num,1))
        for bin in range(self._bin_num):
            self._bin_indicator[bin] = LpVariable(
                name=f"Use_bin_{bin}",
                lowBound = 0,
                upBound = 1,
                cat = LpInteger
            )

        # Initialize item-bin matching
        self._mapping = _init_with_shape(shape=(self._item_num, self._bin_num))
        for item in range(self._item_num):
            for bin in range(self._bin_num):
                self._mapping[item][bin] = LpVariable(
                    name=f"Place_item_{item}_in_bin_{bin}",
                    lowBound=0,
                    upBound=1,
                    cat=LpInteger
                )

    def _add_constraints(self, problem: LpProblem):
        # Mapping limitation: exactly 1 bin for an item
        for item in range(self._item_num):
            problem += (
                lpSum(self._mapping[item][bin] for bin in range(self._bin_num)) == 1,
                f"Mapping_item_{item}_to_1_bin")

        # Capacity limitation: assignments must match capacity constraints
        for bin in range(self._bin_num):
            problem += (
                lpSum(self._mapping[item][bin]*self._future_item_list[item] for item in range(self._item_num)) <= ((self.total_capacity - self.current_capacity[bin])*self._bin_indicator[bin]),
                f"Bin_{bin}_capacity_constraint"
            )
        # Bins with positive current capacity must be counted as used
        for bin in range(self._bin_num):
            if self.current_capacity[bin] > 0:
                problem += (
                    self._bin_indicator[bin] >= 1,
                    f"Bin_{bin}_previous_used"
                )

    def _set_objective(self, problem: LpProblem):
        
        # Adds them together to get number of used bins
        problem += (-1)*lpSum(self._bin_indicator[bin] for bin in range(self._bin_num))

    def _greedy_warm_start(self, problem, future_item_list):
        # Warms start with a simple greedy policy
        current_capacity = np.copy(self.current_capacity)
        if DEBUG: print(f'Warm start current_capacity: {current_capacity}')

        for bin in range(self._bin_num): # updates whether bins are currently used
            if current_capacity[bin] == 0: 
                self._bin_indicator[bin].setInitialValue(0)
            else:
                self._bin_indicator[bin].setInitialValue(1)

        for item in range(self._item_num): # loop over each item
            bin_index = next((i for i,j in enumerate(current_capacity + future_item_list[item] <= self.total_capacity) if j), None)
            # finds the first available bin
            # if DEBUG: print(bin_index, future_item_list[item], current_capacity)
            current_capacity[bin_index] += future_item_list[item] # updates utilization of that bin
            self._bin_indicator[bin_index].setInitialValue(1) # updates values of IP variables
            for bin in range(self._bin_num):
                self._mapping[item][bin].setInitialValue(0)
            self._mapping[item][bin_index].setInitialValue(1)

    
        if DEBUG: print('testing feasibility') # testing feasibility of solution
        for cons in problem.constraints:
            if problem.constraints[cons].valid() == False:
                print('FALSE CONSTRAINT')
                print(cons)



    def _formulate_and_solve(self, future_item_list, warmStart = True, initial_bin = None):
        
        if len(future_item_list) == 0:
            print('Future Item List is Empty!!!!!!')
            return

        assert len(future_item_list) > 0, 'Must be at least one item in future item list to solve'
                
        cur_time = timeit.default_timer()
        self._future_item_list = future_item_list
        self._item_num = len(self._future_item_list)

        if DEBUG: print(f'Number of items total to allocate: {len(future_item_list)}')
        if DEBUG: print(f'Initial bin sizes: {initial_bin}')
        if DEBUG: print(f'Item sizes: {future_item_list}')

        self._get_initial_bin(initial_bin) # gets the initial bin capacity values
        if DEBUG: print(f'Current capacity of bins: {self.current_capacity}')
        if DEBUG: print(f'Total number of bins: {self._bin_num}')
        if DEBUG: print(f'Total number of items: {self._item_num}')
        self._init_variables() # creates variables

        problem = LpProblem( # sets up IP problems
        name=f"Packing_problem",
        sense=LpMaximize
        )

        self._add_constraints(problem=problem) # add constraints and objectives
        self._set_objective(problem=problem)
        

        if DEBUG: print(f"[Timer] {timeit.default_timer() - cur_time:.2f} seconds for set-up.")
        cur_time = timeit.default_timer()
        if warmStart:  # warm starting
            self._greedy_warm_start(problem, future_item_list)
            if DEBUG: print(f"[Timer] {timeit.default_timer() - cur_time:.2f} seconds for warm-start.")
            cur_time = timeit.default_timer()
        self._solver = GUROBI_CMD(msg=0, keepFiles=False, timeLimit = 50, warmStart = warmStart, threads=8, gapAbs = 0.005)
        problem.solve(self._solver) # solves
        if DEBUG: print(f"[Timer] {timeit.default_timer() - cur_time:.2f} seconds for solution.")
        cur_time = timeit.default_timer()

        if LpStatus[problem.status] != "Optimal":
            print(future_item_list)
            print(initial_bin)
            print('Found infeasible problem')
        if DEBUG:   print(f'[Objective Value] {problem.objective.value()}')
        return problem.objective.value() + np.sum(initial_bin) # returns objective value
