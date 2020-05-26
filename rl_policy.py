import os
from baselines.common import tf_util as U
from baselines import logger
from baselines.common import set_global_seeds
import gym
import gym_trafficnetwork
from baselines.bench import Monitor
import numpy as np

sim_duration = 6.0 # hours
P = 3
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


env = gym.make('ParallelNetwork-v0')
env.set('sim_duration', sim_duration) # hours
env.set('start_empty', False)
env.set('start_from_equilibrium', False)
env.set('P', P)
env.set('init_learn_rate', 0.5)
env.set('constant_learn_rate', True)
env.set('accident_param', accident_param) # expected number of accidents in 1 hour
# If you change the two params below, you should change the model_path below or retrain a model.
env.set('demand', [1.993974,2.990961]) # human-driven and autonomous cars per second, respectively
env.set('demand_noise_std', [0.1993974,0.2990961]) # human-driven and autonomous cars per second, respectively

filename = 'RoadNetworkP' + str(P) + 'Accidents' + str(1 if accident_param > 0 else 0)

model_path = os.path.join('trained_models', filename)
pi = load_policy(env)
U.load_state(model_path)

env.seed(1909) # Use seed 1909 for the results shown in Fig. 6 with the above default parameters

o_vals = []
o = env.reset() # reset is compulsory, don't assume the constructor calls it.
r_vals = []
o_vals.append(o)
d = False
aut_distribution = np.array([1.0/env.num_roads]*env.num_roads)
n_t_a = env.init_learn_rate
t = 0
while not d:
	a = pi.act(stochastic=False, ob=o)[0]
	o,r,d,_ = env.step(a)
	o_vals.append(o)
	r_vals.append(r)
	t += 1
	print('At time step ' + str(t) + ', there are ' + str(-np.sum(r_vals)) + ' cars in the system.')
