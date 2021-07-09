import os
from baselines.common import tf_util as U
from baselines import logger
from baselines.common import set_global_seeds
import gym
import gym_trafficnetwork
from baselines.bench import Monitor
import numpy as np

sim_duration = 6.0 # hours
network_type = 'multiOD' # type 'parallel' or 'general' or 'multiOD'
P = 3 # number of paths (only for parallel -- the general network graph is defined inside its environment file)
accident_param = 0.6 # expected number of accidents in 1 hour

def load_policy(env):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=256, num_hid_layers=2)
    workerseed = 0
    set_global_seeds(workerseed)
    env_max_step_size = env.max_step_size
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(0)))
    env.seed(0)

    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=1,
            timesteps_per_actorbatch=env_max_step_size,
            clip_param=0.2, entcoeff=0.005,
            optim_epochs=5,
            optim_stepsize=3e-4,
            optim_batchsize=1,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
        )
    env.close()

    return pi


if network_type.lower() == 'parallel':
    filename = 'ParallelNetworkP' + str(P) + 'Accidents' + str(1 if accident_param > 0 else 0)
    env = gym.make('ParallelNetwork-v0')
elif network_type.lower() == 'general':
    filename = 'GeneralNetworkAccidents' + str(1 if accident_param > 0 else 0)
    env = gym.make('GeneralNetwork-v0')
elif network_type.lower() == 'multiod':
    filename = 'GeneralNetworkMultiODAccidents' + str(1 if accident_param > 0 else 0)
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
# If you change the two params below, you should change the model_path below or retrain a model.
env.set('demand', [[[0.346,0.519],[0.346,0.519]],[[0.346,0.519],[0.346,0.519]]]) # human-driven and autonomous cars per second, respectively
env.set('demand_noise_std', [[[0.0346,0.0519],[0.0346,0.0519]],[[0.0346,0.0519],[0.0346,0.0519]]]) # human-driven and autonomous cars per second, respectively


model_path = os.path.join('trained_models', filename)
pi = load_policy(env)
U.load_state(model_path)

env.seed(17)

o_vals = []
o = env.reset() # reset is compulsory, don't assume the constructor calls it.
r_vals = []
o_vals.append(o)
d = False
aut_distribution = env.aut_distribution.copy()
aut_distribution2 = np.concatenate([x for y in aut_distribution for x in y])
n_t_a = env.init_learn_rate
t = 0
while not d:
    a = pi.act(stochastic=False, ob=o)[0]
    o,r,d,_ = env.step(a)
    o_vals.append(o)
    r_vals.append(r)
    t += 1
    print('At time step ' + str(t) + ', there are ' + str(-np.sum(r_vals)) + ' cars in the system.')
