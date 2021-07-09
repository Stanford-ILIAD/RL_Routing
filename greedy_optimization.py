import gym
import gym_trafficnetwork
import numpy as np
import copy
import scipy.optimize as opt
from multiprocessing import Pool
import os
import time
from itertools import product

def func(x, env):
    temp_env = copy.deepcopy(env)
    o,r,d,_ = temp_env.step(x)
    temp_latencies = temp_env.measure_latencies()
    latencies = np.zeros(temp_env.num_paths)
    for o_id in range(2):
        for d_id in range(2):
            latencies[temp_env.path_ids_od[o_id][d_id]] = temp_latencies[o_id][d_id]
    return np.sum([((temp_env.cells[c_id].state * temp_env.cells[c_id].mu) * latencies.reshape((-1,1))).sum() for c_id in range(temp_env.num_cells)])


def optimize(env, num_winners=2): # optimize with parallelization & a time constraint
    start_t = time.time()
    cpu_count = os.cpu_count()
    xs = [env.action_space.sample() for _ in range(cpu_count)]
    dim = len(xs[0])
    p = Pool(cpu_count)
    opt_t = 0
    min_R = np.Inf
    min_x = xs[0] # just in case..
    while time.time() - start_t < 60.:
        if opt_t > 0:
            winner_xs = np.array(xs)[np.argsort(R)[:num_winners]]
            xs = []
            populated = False
            while not populated:
                for winner_id in range(len(winner_xs)):
                    xs.append(winner_xs[winner_id] + np.random.randn(dim)*0.5)
                    if len(xs) == cpu_count:
                        populated = True
                        break
        all_R = p.starmap(func, product(xs, [env]))
        R = np.asarray(all_R)
        opt_t += 1
        minnow = np.min(R)
        if minnow < min_R:
            min_R = minnow
            min_x = xs[np.argmin(R)]
    print(min_R)
    return min_x


if __name__ == '__main__':  

    sim_duration = 6.0 # hours
    network_type = 'multiOD' # type 'parallel' or 'general' or 'multiOD'
    P = 3 # number of paths (only for parallel -- the general network graph is defined inside its environment file)
    accident_param = 0.6 # expected number of accidents in 1 hour

    if network_type.lower() == 'parallel':
        env = gym.make('ParallelNetwork-v0')
    elif network_type.lower() == 'general':
        env = gym.make('GeneralNetwork-v0')
    elif network_type.lower() == 'multiod':
        env = gym.make('GeneralNetworkMultiOD-v0')
    else:
        assert False, 'network_type is invalid.'

    env.set('sim_duration', sim_duration) # hours
    env.set('start_empty', False)
    env.set('start_from_equilibrium', False)
    if network_type.lower() == 'parallel': env.set('P', P)
    env.set('init_learn_rate', 0.5)
    env.set('constant_learn_rate', True)
    env.set('accident_param', accident_param) # expected number of accidents in 1 hour
    env.set('demand', [[[0.346,0.519],[0.346,0.519]],[[0.346,0.519],[0.346,0.519]]]) # human-driven and autonomous cars per second, respectively
    env.set('demand_noise_std', [[[0.0346,0.0519],[0.0346,0.0519]],[[0.0346,0.0519],[0.0346,0.0519]]]) # human-driven and autonomous cars per second, respectively
    env.seed(4)
    print('Environment is set!')

        
    o_vals = []
    o = env.reset() # reset is compulsory, don't assume the constructor calls it.
    r_vals = []
    o_vals.append(o)
    d = False
    t = 0
    while not d:
        x = optimize(env)
        o,r,d,_ = env.step(x)
        o_vals.append(o)
        r_vals.append(r)
        t += 1
        print('At time step ' + str(t) + ', there are ' + str(-np.sum(r_vals)) + ' cars in the system.')
