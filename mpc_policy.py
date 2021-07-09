import gym
import gym_trafficnetwork
import numpy as np
import copy
import scipy.optimize as opt

mpc_receding_horizon = 4
mpc_planning_period = 1
sample_count = 12

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
env.initialize()
env.seed(0)
print('Environment is set!')

def mpc_func(action, *args):
    arr = []
    for _ in range(sample_count):
        copy_env = copy.deepcopy(args[0])
        action_dim = copy_env.action_space.shape[0]
        action = np.reshape(action, [-1,action_dim])
        total_rew = 0.
        for act in action:
            _, r, d, _ = copy_env.step(act)
            total_rew += r
            if d: break
        arr.append(-total_rew)
    return np.mean(arr)

def mpc_policy(env):
    z = env.action_space.shape[0] * mpc_receding_horizon
    lb = np.array(list(env.action_space.low) * mpc_receding_horizon)
    ub = np.array(list(env.action_space.high) * mpc_receding_horizon)
    opt_res = opt.fmin_l_bfgs_b(mpc_func, x0=np.random.uniform(low=lb, high=ub), args=(env,), bounds=[(lbx, ubx) for lbx,ubx in zip(lb, ub)], approx_grad=True)
    opt_res = np.reshape(opt_res[0], [-1,env.action_space.shape[0]])
    return opt_res[:mpc_planning_period]
    
o_vals = []
o = env.reset() # reset is compulsory, don't assume the constructor calls it.
r_vals = []
o_vals.append(o)
d = False
t = 0
mpc_counter = mpc_planning_period
while not d:
    if mpc_counter == mpc_planning_period:
        mpc_action = mpc_policy(env)
        mpc_counter = 0
    action = mpc_action[mpc_counter]
    mpc_counter += 1
    o,r,d,_ = env.step(action)
    o_vals.append(o)
    r_vals.append(r)
    t += 1
    print('At time step ' + str(t) + ', there are ' + str(-np.sum(r_vals)) + ' cars in the system.')
