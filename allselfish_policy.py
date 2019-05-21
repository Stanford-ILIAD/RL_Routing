import gym
import gym_roadnetwork
import numpy as np

env = gym.make('RoadNetwork-v0')
env.set('sim_duration', 6.0) # hours
env.set('start_empty', False)
env.set('start_from_equilibrium', False)
env.set('P', 3)
env.set('init_learn_rate', 0.5)
env.set('constant_learn_rate', True)
env.set('accident_param', 0.6) # expected number of accidents in 1 hour
env.set('demand', [1.993974,2.990961]) # human-driven and autonomous cars per second, respectively
env.set('demand_noise_std', [0.1993974,0.2990961]) # human-driven and autonomous cars per second, respectively
env.seed(1909) # Use seed 1909 for the results shown in Fig. 6 with the above default parameters
print('Environment is set!')


o_vals = []
o = env.reset() # reset is compulsory, don't assume the constructor calls it.
r_vals = []
o_vals.append(o)
d = False
aut_distribution = np.array([1.0/env.num_roads]*env.num_roads)
n_t_a = env.init_learn_rate
t = 0
while not d:
	aut_distribution, n_t_a = env.set_selfish_decision(aut_distribution, n_t_a)
	o,r,d,_ = env.step(aut_distribution)
	o_vals.append(o)
	r_vals.append(r)
	t += 1
	print('At time step ' + str(t) + ', there are ' + str(-np.sum(r_vals)) + ' cars in the system.')
