import gym
import gym_trafficnetwork
import numpy as np
import copy
import scipy.optimize as opt

mpc_receding_horizon = 5
mpc_planning_period = 1

env = gym.make('GeneralNetwork-v0')
env.set('sim_duration', 6.0) # hours
env.set('start_empty', False)
env.set('start_from_equilibrium', False)
#env.set('P', 3) # only for ParallelNetwork
env.set('init_learn_rate', 0.5)
env.set('constant_learn_rate', True)
env.set('accident_param', 0.6) # expected number of accidents in 1 hour
env.set('demand', [1.0,1.5]) # human-driven and autonomous cars per second, respectively
env.set('demand_noise_std', [0.1,0.15]) # human-driven and autonomous cars per second, respectively
env.seed(1909) # Use seed 1909 for the results shown in Fig. 6 with the above default parameters
print('Environment is set!')


def mpc_func(action, *args):
    copy_env = copy.deepcopy(args[0])
    action_dim = copy_env.action_space.shape[0]
    action = np.reshape(action, [-1,action_dim])
    total_rew = 0.
    for act in action:
        _, r, d, _ = copy_env.step(act)
        total_rew += r
        if d: break
    return -total_rew

def mpc_policy(env):
    z = env.action_space.shape[0] * mpc_receding_horizon
    lb = np.array(list(env.action_space.low) * mpc_receding_horizon)
    ub = np.array(list(env.action_space.high) * mpc_receding_horizon)
    #import pdb; pdb.set_trace()
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
