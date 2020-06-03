import gym
import gym_trafficnetwork
import numpy as np


sim_duration = 6.0 # hours
network_type = 'parallel' # type 'parallel' or 'general'
P = 3 # number of paths (only for parallel -- the general network graph is defined inside its environment file)
accident_param = 0.6 # expected number of accidents in 1 hour

if network_type.lower() == 'parallel':
    env = gym.make('ParallelNetwork-v0')
elif network_type.lower() == 'general':
    env = gym.make('GeneralNetwork-v0')
else:
    assert False, 'network_type is invalid.'

env.set('sim_duration', sim_duration) # hours
env.set('start_empty', False)
env.set('start_from_equilibrium', False)
if network_type.lower() == 'parallel': env.set('P', P)
env.set('init_learn_rate', 0.5)
env.set('constant_learn_rate', True)
env.set('accident_param', accident_param) # expected number of accidents in 1 hour
env.set('demand', [1.993974,2.990961]) # human-driven and autonomous cars per second, respectively
env.set('demand_noise_std', [0.1993974,0.2990961]) # human-driven and autonomous cars per second, respectively
env.seed(17)
print('Environment is set!')


o_vals = []
o = env.reset() # reset is compulsory, don't assume the constructor calls it.
r_vals = []
o_vals.append(o)
d = False
aut_distribution = np.array([1.0/env.num_paths]*env.num_paths)
n_t_a = env.init_learn_rate
t = 0
while not d:
	aut_distribution, n_t_a = env.set_selfish_decision(aut_distribution, n_t_a)
	o,r,d,_ = env.step(aut_distribution)
	o_vals.append(o)
	r_vals.append(r)
	t += 1
	print('At time step ' + str(t) + ', there are ' + str(-np.sum(r_vals)) + ' cars in the system.')
