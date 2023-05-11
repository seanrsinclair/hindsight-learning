import numpy as np
import pulp
import itertools
from guidedrl.bin_packing.optimal_bin_packing import BinPackingILP
from guidedrl.bin_packing.binpacking import BIG_NEG_REWARD


DEBUG = False


def get_exogenous_dataset(env, dataset_size):
    return np.asarray([env.get_item() for _ in range(dataset_size)])



def get_optimal_policy(env):
    state_limits = np.copy(env.observation_space.high + 1)
    # adding one to the max value of observation space so that np.arange() enumerates over all values
    
    qVals = np.zeros(np.append(state_limits, [env.step_limit, env.bin_capacity]))
    vVals = np.zeros(np.append(state_limits, env.step_limit))
    # setting up arrays for the qVals and vVals, indexed by s,h,a
    if DEBUG:
        print(f'State Limits: {state_limits}')
        print(f'qVals Shape: {qVals.shape}')
        print(f'vVals Shape: {vVals.shape}')
    for h in np.arange(env.step_limit-1, -1, -1): # loops backwards in time
        print(f'Step: {h}')
        for s in itertools.product(*[np.arange(x) for x in state_limits]): # loops over each state
            if np.sum(list(s)[:-1]) <= h and list(s)[-1] in env.item_sizes: # in a valid state
                    # dictated by fewer bins opened than current step, and the last component corresponds
                    # to an item that is contained in the item size list

                cur_item_size = list(s)[-1] # gets out the current item
                for a in range(env.bin_capacity): # loops over all actions
                    if a == 0: # opening up a new bin
                        immediate_reward = -1 # penalized by one for opening up a new bin
                        for index in range(len(env.item_sizes)): # calculating expectation of V^(s_new)
                            size = env.item_sizes[index]
                            prob = env.item_probs[index]
                            # Calculating new state:
                            list_s = list(s)
                            list_s[cur_item_size-1] += 1 # create new bin of current items size
                            list_s[-1] = size # update current arrival
                            new_s = tuple(list_s) # converts list back to tuple and adds on to q value
                            qVals[s+(h,)+(a,)] += prob*(immediate_reward + (0 if h == env.step_limit - 1 else vVals[new_s+(h+1,)]))
                            
                    elif a > 0: # adding item to an existing box
                        if s[a-1] == 0: # no bin exists at that level, so infeasible action
                            immediate_reward = BIG_NEG_REWARD
                            qVals[s+(h,)+(a,)] += immediate_reward
                        elif cur_item_size + a > env.bin_capacity: # invalid action as we are overfilling a bin
                            immediate_reward = BIG_NEG_REWARD
                            qVals[s+(h,)+(a,)] += immediate_reward
                        else:
                            # adding item to an existing bin
                            immediate_reward = 0
                            for index in range(len(env.item_sizes)): # calculating expectation of V^(s_new)
                                size = env.item_sizes[index]
                                prob = env.item_probs[index]
                                list_s = list(s)
                                list_s[a-1] -= 1 # eliminate old bin at level
                                list_s[a+cur_item_size-1] += 1 # update bin to the correct level
                                list_s[-1] = size # update current arrival
                                new_s = tuple(list_s)
                                qVals[s+(h,)+(a,)] += prob*(immediate_reward + (0 if h == env.step_limit - 1 else vVals[new_s+(h+1,)]))
            vVals[s+(h,)] = np.max([qVals[s+(h,)+(a,)] for a in range(env.bin_capacity)]) # updating v value as max of q value
    return vVals, qVals


def get_ip_values(env):
    
    state_limits = np.copy(env.observation_space.high+1)
    # adding one to the max value of observation space so that np.arange() enumerates over all values

    qIP = np.zeros(np.append(state_limits, [env.step_limit, env.bin_capacity]))
    vIP = np.zeros(np.append(state_limits, env.step_limit))
    # setting up the matrices to hold the final values
    binIP = BinPackingILP(env)
    # initializing the IP solver
    
    for h in np.arange(env.step_limit - 1, -1, -1): # loops back across h
        print(f'Step: {h}')
        for s in itertools.product(*[np.arange(x) for x in state_limits]): # loops over each state
            cur_item_size = list(s)[-1]
            if np.sum(list(s)[:-1]) <= h and list(s)[-1] in env.item_sizes: # in a valid state
                cur_item_size = list(s)[-1] # gets out the current item size
                if DEBUG: print(f'Current item size: {cur_item_size}')
                for a in range(env.bin_capacity): # loops over all possible actions
                    if a == 0: # opening up new bin
                        if h == env.step_limit - 1:
                            qIP[s+(h,)+(a,)] = -1 # final reward is -1 for this newly opened bin
                        else:
                            new_s = list(s)
                            new_s[cur_item_size-1] += 1 # updates the current state
                            if DEBUG: print(f'New state: {new_s}')
                            for inds in itertools.product(*[np.arange(len(env.item_sizes)) for _ in range(env.step_limit-(h+1))]):
                                future_item_list = [env.item_sizes[i] for i in inds] 
                                prob = np.product([env.item_probs[i] for i in inds])
                                qIP[s+(h,)+(a,)] += prob * (-1 + binIP._formulate_and_solve(future_item_list, warmStart= True, initial_bin = new_s[:-1]))
                    
                    elif a > 0 and s[a-1] == 0: # infeasible action, no bin open at that level
                        qIP[s+(h,)+(a,)] = BIG_NEG_REWARD
                    elif a > 0 and cur_item_size + a > env.bin_capacity: # infeasible action, overflowing bin
                        qIP[s+(h,)+(a,)] = BIG_NEG_REWARD
                    else: # adding item to existing bin
                        if h == env.step_limit - 1:
                            qIP[s+(h,)+(a,)] = 0
                        else:
                            new_s = list(s)
                            new_s[a-1] -= 1 # updates state, again does the monte carlo for the future rewards expectation
                            new_s[a + cur_item_size-1] += 1
                            for inds in itertools.product(*[np.arange(len(env.item_sizes)) for _ in range(env.step_limit-(h+1))]):
                                future_item_list = [env.item_sizes[i] for i in inds] 
                                prob = np.product([env.item_probs[i] for i in inds])
                                qIP[s+(h,)+(a,)] += prob * binIP._formulate_and_solve(future_item_list, warmStart= True, initial_bin = new_s[:-1])

            vIP[s+(h,)] = np.max(qIP[s+(h,)]) # updates the value of the "IP" policy
    return vIP, qIP