import gym
import gym_trafficnetwork
import numpy as np


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
	o,r,d,_ = env.step(env.action_space.sample())
	o_vals.append(o)
	r_vals.append(r)
	t += 1
	print('At time step ' + str(t) + ', there are ' + str(-np.sum(r_vals)) + ' cars in the system.')
