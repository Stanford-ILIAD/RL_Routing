import os
from baselines.common import tf_util as U
from baselines import logger
from baselines.common import set_global_seeds
import gym
import gym_trafficnetwork
from mpi4py import MPI
from baselines.bench import Monitor

sim_duration = 5.0 # hours
network_type = 'parallel' # type 'parallel' or 'general'
P = 3 # number of paths (only for parallel -- the general network graph is defined inside its environment file)
accident_param = 0.6 # expected number of accidents in 1 hour

def train(env, seed, model_path=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=256, num_hid_layers=2)
    rank = MPI.COMM_WORLD.Get_rank()
    workerseed = seed + 10000 * rank if seed is not None else None
    set_global_seeds(workerseed)
    env_max_step_size = env.max_step_size
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=4e7,
            timesteps_per_actorbatch=4*env_max_step_size, # we used 32 CPUs
            clip_param=0.2, entcoeff=0.005,
            optim_epochs=5,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
        )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi


def main():
    logger.configure()
    if network_type.lower() == 'parallel':
        filename = 'ParallelNetworkP' + str(P) + 'Accidents' + str(1 if accident_param > 0 else 0)
        env = gym.make('ParallelNetwork-v0')
    elif network_type.lower() == 'general':
        filename = 'GeneralNetworkAccidents' + str(1 if accident_param > 0 else 0)
        env = gym.make('GeneralNetwork-v0')
    else:
        assert False, 'network_type is invalid.'
        
    model_path = os.path.join('trained_models', filename)
    
    env.set('sim_duration', sim_duration) # hours
    env.set('start_empty', False)
    env.set('start_from_equilibrium', False)
    if network_type.lower() == 'parallel': env.set('P', P)
    env.set('init_learn_rate', 0.5)
    env.set('constant_learn_rate', True)
    env.set('accident_param', accident_param) # expected number of accidents in 1 hour
    env.set('demand', [1.993974,2.990961]) # human-driven and autonomous cars per second, respectively
    env.set('demand_noise_std', [0.1993974,0.2990961]) # human-driven and autonomous cars per second, respectively
    
    # train the model
    train(env, seed=None, model_path=model_path)

if __name__ == '__main__':
    main()
