import numpy as np

seed = 387
num_rounds = 100
num_accepts = int((3/5)*num_rounds)
num_types = 4
eps = .5
np.set_printoptions(precision=2, suppress=True)
rg = np.random.default_rng(seed)

arrival_probs = np.zeros((num_rounds,num_types))
arrival_start = rg.random(size=num_types) * 2 * np.pi
arrival_frequency = rg.random(size=num_types) * 0.25 * np.pi
for i in range(num_types):
    arrival_probs[:,i] = np.arange(arrival_start[i], arrival_start[i]+num_rounds*arrival_frequency[i]+eps, arrival_frequency[i])[0:num_rounds]
arrival_probs = 1 + rg.random(size=num_types) + np.sin(arrival_probs)
row_sums = np.sum(arrival_probs, axis=1)
arrival_probs = np.divide(arrival_probs, row_sums[:,None])

abilities = np.linspace(1.0/num_types, 1.0, num=num_types)


def create_policy(num_rounds, num_accepts, num_types):
    policy = np.empty((num_rounds,num_accepts+1,num_types))
    policy.fill(np.nan)
    policy[:,0,:]=0
    for j in range(1,num_accepts+1):
        policy[0:j,j:,:]=1
    return policy


def evaluate_policy(policy, arrival_probs, abilities):
    num_rounds = np.shape(policy)[0]
    num_accepts = np.shape(policy)[1]-1
    num_types = np.shape(policy)[2]
    values = np.zeros(np.shape(policy))
    values[0,1:,:] = abilities
    for t in range(1,num_rounds):
        next_round_arrivals = arrival_probs[num_rounds-t,:]
        for b in range(1,num_accepts+1):
            current_policy = policy[t,b,:]
            accept_scores = abilities + np.dot(next_round_arrivals, values[t-1, b-1, :])
            reject_score = np.dot(next_round_arrivals, values[t-1, b, :])
            
            values[t,b,:] = reject_score * (1.0 - current_policy) + np.multiply(current_policy, accept_scores)
    return values


def hindsight_optimal(future_arrivals, abilities, budget):
    num_types = np.shape(abilities)[0]
    value = 0.0
    for j in range(num_types):
        index = num_types-j-1
        if budget <= future_arrivals[index]:
            value += abilities[index]*budget
            break
        else:
            value += abilities[index]*future_arrivals[index]
            budget -= future_arrivals[index]
    return value

    
pi_star = create_policy(num_rounds,num_accepts,num_types)
optimal_values = np.zeros(np.shape(pi_star))
optimal_values[0,1:,:] = abilities
for t in range(1,num_rounds):
    next_round_arrivals = arrival_probs[num_rounds-t,:]
    for b in range(1,num_accepts+1):
        accept_scores = abilities + np.dot(next_round_arrivals, optimal_values[t-1, b-1, :])
        reject_score = np.dot(next_round_arrivals, optimal_values[t-1, b, :])
        optimal_values[t,b,:]=np.maximum(accept_scores,reject_score)
        pi_star[t,b,:]=accept_scores>reject_score
        
print("V_star", np.dot(optimal_values[-1,-1,:], arrival_probs[0,:]))

training_trace = np.zeros((num_rounds,), dtype=int)
empirical_arrivals = np.zeros((num_rounds,num_types))
for i in range(num_rounds):
    training_trace[i] = rg.choice(num_types,1,p=arrival_probs[i,:])[0]
for i in range(num_rounds):
    empirical_arrivals[:i+1,training_trace[i]] += 1
    
print("Trace", training_trace)

pi_dagger = create_policy(num_rounds,num_accepts,num_types)
Q_dagger_accepts = np.zeros(np.shape(pi_dagger))
Q_dagger_rejects = np.zeros(np.shape(pi_dagger))
Q_dagger_accepts[0,1:,:] = abilities
for t in range(1,num_rounds):
    remainder = empirical_arrivals[num_rounds-t,:]
    for b in range(1,num_accepts+1):
        Q_dagger_rejects[t,b,:] = hindsight_optimal(remainder, abilities, b)
        Q_dagger_accepts[t,b,:] = abilities + hindsight_optimal(remainder, abilities, b-1)
        pi_dagger[t,b,:] = Q_dagger_accepts[t,b,:] > Q_dagger_rejects[t,b,:]
        
V_dagger = evaluate_policy(pi_dagger, arrival_probs, abilities)
print("V_dagger", np.dot(V_dagger[-1,-1,:], arrival_probs[0,:]))
Optimistic_V = np.maximum(Q_dagger_accepts[-1,-1,:], Q_dagger_rejects[-1,-1,:])
print("Optimistic_V", np.max(V_dagger[-1,-1,:]))

random_policy = create_policy(num_rounds,num_accepts,num_types)
random_policy = np.nan_to_num(random_policy, nan=0.5)
V_random = evaluate_policy(random_policy, arrival_probs, abilities)
print("V_random", np.dot(V_random[-1,-1,:], arrival_probs[0,:]))

lazy_policy = create_policy(num_rounds,num_accepts,num_types)
lazy_policy = np.nan_to_num(lazy_policy, nan=0)
V_lazy = evaluate_policy(lazy_policy, arrival_probs, abilities)
print("V_lazy", np.dot(V_lazy[-1,-1,:], arrival_probs[0,:]))

eager_policy = create_policy(num_rounds,num_accepts,num_types)
eager_policy = np.nan_to_num(eager_policy, nan=1)
V_eager = evaluate_policy(eager_policy, arrival_probs, abilities)
print("V_eager", np.dot(V_eager[-1,-1,:], arrival_probs[0,:]))

hl_policy = create_policy(num_rounds,num_accepts,num_types)
hl_policy = np.nan_to_num(hl_policy, nan=0.5)
for epochs in range(100000):
    remaining_budget = num_accepts
    for i in range(num_rounds):
        current_action = 0.5
        if current_action > 0 and current_action < 1:
            if rg.uniform() < current_action:
                current_action = 1
            else:
                current_action = 0
        #Imitate Q_dagger on current state
        accept_score = Q_dagger_accepts[num_rounds-i-1, remaining_budget, training_trace[i]]
        reject_score = Q_dagger_rejects[num_rounds-i-1, remaining_budget, training_trace[i]]
        if accept_score > reject_score:
            hl_policy[num_rounds-i-1, remaining_budget, training_trace[i]] = 1
        elif accept_score < reject_score:
            hl_policy[num_rounds-i-1, remaining_budget, training_trace[i]] = 0
        else:
            hl_policy[num_rounds-i-1, remaining_budget, training_trace[i]] = 0.5
        
        #Also update for arrivals other than training_trace[i]
        for j in range(num_types):
            accept_score = Q_dagger_accepts[num_rounds-i-1, remaining_budget, j]
            reject_score = Q_dagger_rejects[num_rounds-i-1, remaining_budget, j]
            if accept_score > reject_score:
                hl_policy[num_rounds-i-1, remaining_budget, j] = 1
            elif accept_score < reject_score:
                hl_policy[num_rounds-i-1, remaining_budget, j] = 0
            else:
                hl_policy[num_rounds-i-1, remaining_budget, j] = 0.5
            
        if current_action == 1:
            remaining_budget -= 1
        
V_hl = evaluate_policy(hl_policy, arrival_probs, abilities)
print("V_hl", np.dot(V_hl[-1,-1,:], arrival_probs[0,:]))

q_policy = create_policy(num_rounds,num_accepts,num_types)
q_policy = np.nan_to_num(q_policy, nan=0.5)
Q_accepts = np.zeros(np.shape(q_policy))
Q_rejects = np.zeros(np.shape(q_policy))
learning_rate = 0.1
scratch_Q_accepts = np.copy(Q_accepts)
scratch_Q_rejects = np.copy(Q_rejects)
Q_accepts[0,1:,:] = abilities
for epochs in range(100000):
    remaining_budget = num_accepts
    scratch_Q_rejects[:] = 0
    scratch_Q_accepts[:] = 0
    for i in range(num_rounds-1):
        current_action = 0.5
        if current_action > 0 and current_action < 1:
            if rg.uniform() < current_action:
                current_action = 1
            else:
                current_action = 0
                
        #REJECT
        if current_action == 0:
            scratch_Q_rejects[num_rounds-i-1, remaining_budget, training_trace[i]] = \
                Q_rejects[num_rounds-i-1, remaining_budget, training_trace[i]] + \
                learning_rate * (np.maximum(Q_accepts[num_rounds-i-2, remaining_budget, training_trace[i+1]], \
                                            Q_rejects[num_rounds-i-2, remaining_budget, training_trace[i+1]]) \
                                - Q_rejects[num_rounds-i-1, remaining_budget, training_trace[i]])
        #ACCEPT
        else:
            scratch_Q_accepts[num_rounds-i-1, remaining_budget, training_trace[i]] = \
                Q_accepts[num_rounds-i-1, remaining_budget, training_trace[i]] + \
                learning_rate * (abilities[training_trace[i]] + \
                                np.maximum(Q_accepts[num_rounds-i-2, remaining_budget-1, training_trace[i+1]], \
                                            Q_rejects[num_rounds-i-2, remaining_budget-1, training_trace[i+1]]) \
                                - Q_accepts[num_rounds-i-1, remaining_budget, training_trace[i]])
            
        
        #Also update for arrivals other than training_trace[i]
        for j in range(num_types):
            if j == training_trace[i]:
                continue
                
            #REJECT
            if current_action == 0:
                scratch_Q_rejects[num_rounds-i-1, remaining_budget, j] = \
                    Q_rejects[num_rounds-i-1, remaining_budget, j] + \
                    learning_rate * (np.maximum(Q_accepts[num_rounds-i-2, remaining_budget, training_trace[i+1]], \
                                            Q_rejects[num_rounds-i-2, remaining_budget, training_trace[i+1]]) \
                                - Q_rejects[num_rounds-i-1, remaining_budget, j])
            #ACCEPT
            else:
                scratch_Q_accepts[num_rounds-i-1, remaining_budget, j] = \
                    Q_accepts[num_rounds-i-1, remaining_budget, j] + \
                    learning_rate * (abilities[j] + \
                                np.maximum(Q_accepts[num_rounds-i-2, remaining_budget-1, training_trace[i+1]], \
                                            Q_rejects[num_rounds-i-2, remaining_budget-1, training_trace[i+1]]) \
                                - Q_accepts[num_rounds-i-1, remaining_budget, j])
            
        if current_action == 1:
            remaining_budget -= 1
            
    accept_indices = (scratch_Q_accepts != 0)
    Q_accepts[accept_indices] = scratch_Q_accepts[accept_indices]
    reject_indices = (scratch_Q_rejects != 0)
    Q_rejects[reject_indices] = scratch_Q_rejects[reject_indices]

for t in range(1,num_rounds):
    for b in range(1,num_accepts+1):
        q_policy[t,b,:] = Q_accepts[t,b,:] > Q_rejects[t,b,:]
            
V_q = evaluate_policy(q_policy, arrival_probs, abilities)
print("V_q", np.dot(V_q[-1,-1,:], arrival_probs[0,:]))
