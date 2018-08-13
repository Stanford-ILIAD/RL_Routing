from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from road_network import RoadNetwork

def train(num_timesteps, seed):
    from baselines.ppo1 import pposgd_simple, mlp_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    env = RoadNetwork(num_roads=2, vfkph=[50,100], num_cells=[3,5], cell_dist_meters=[500,1000])
    
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=256, num_hid_layers=3)

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=256,
        clip_param=0.2, entcoeff=0.00,
        optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=128,
        gamma=0.99, lam=0.95, schedule='linear'
    )
    env.close()

def main():
    train(num_timesteps=2e6, seed=4)

if __name__ == '__main__':
    main()