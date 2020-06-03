import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque
import copy
import sys


MACHINE_PRECISION = 1e-10

secperhr = 60*60
kmpermiles = 1.609344
quantization = 1. # the more this value the more quantization we get
T = 1.0/60*secperhr*quantization # seconds per time step (1/60 of an hour to travel 1 mile at 60mph)
tauh = 2.0 # human reaction time
taua = 1.0 # aut reaction time
L = 4.0 # car length (meters)
MANY_ITERS = 5000 # if estimated latency is longer than this consider it to be forever. clip estimated number of iterations here


class Cell:
    def __init__(self, vfkph, cell_length, num_lanes, priority):
        self.vfms = vfkph*1000.0/secperhr # free-flow speed in meters per second
        self.headway = [self.vfms * tauh, self.vfms * taua] # meters
        self.cell_length = cell_length # meters
        assert priority > MACHINE_PRECISION, 'cell priorities must be strictly positive'
        self.priority = priority
        if not np.isinf(num_lanes): # so that the cell is not the queue -- no accidents in the queue
            self.accidents = np.array([0]*num_lanes) # nonpositive values will mean the lane is clear. Positive values will be the number of timesteps required to clear.
        else:
            self.accidents = np.array([])
        self.vf = T * self.vfms / self.cell_length # free-flow speed in cells per time step
        self.nj = self.num_active_lanes * self.cell_length / L # jam density (vehicles per cell)

        self.reset(num_paths=0)# temporarily 0

    @property
    def num_active_lanes(self):
        if self.accidents.size > 0: # not the queue
            return np.sum(self.accidents <= MACHINE_PRECISION)
        return np.inf
        
    @property
    def num_lanes(self):
        if self.accidents.size > 0: # not the queue
            return len(self.accidents)
        return np.inf
        
    @property
    def autonomy(self):
        n = self.n
        if not np.isclose(n,0):
            return self.state[1] / n
        return 0.5 # so that we won't get divide by 0 errors
        
    @property
    def n(self):
        assert np.all(-MACHINE_PRECISION <= self.state)
        return self.state.sum()
        
    def reset(self, num_paths):
        self.state = np.zeros(2) # nh, na
        self.mu = np.zeros((num_paths,2))
        self.accidents *= 0 # clear all accidents
        self.update_params()
        
    def randomize_state(self, downstream_path_ids):
        self.mu[downstream_path_ids,:] = self.np_random.rand(len(downstream_path_ids),2)
        self.mu /= self.mu.sum(axis=0)
        aut = self.np_random.rand()
        n = self.np_random.rand() * 1.2 * self.critdens(aut)
        self.state = np.array([(1-aut) * n, aut * n])
        self.update_params()
    
    # creates an accident with probability p and makes sure the cell is not completely blocked
    def simulate_accident(self, p=5e-6):
        if self.accidents.size > 0: # not the queue 
            self.accidents -= 1
            if self.state.sum() > 1 + MACHINE_PRECISION: # no accidents if there are fewer than 2 cars
                for i in range(self.num_lanes):
                    if self.np_random.rand() < p and not (self.accidents[i] <= 0 and self.num_active_lanes == 1): # makes sure we are not entirely blocking the road
                        # Based on https://www.denenapoints.com/long-usually-take-clear-accident-scene-houston/
                        # The average time to clear an accident was about 33-34 minutes in 2011 in Houston. Let's say it is now 30 minutes = 0.5 hours.
                        self.accidents[i] += self.np_random.poisson(lam=0.5*secperhr/T) # so that we allow for a second accident in the same lane
            self.accidents = np.clip(self.accidents, 0, None) # let's not make it go to negative for simpler observation model for RL
    
    # calculate the relevant parameters, sending and receiving functions for the cell (depends on autonomy level)
    def update_params(self):
        self.nc = self.critdens(self.autonomy)
        self.w = self.shockspeed(self.nc)
        self.cap = self.maxflow(self.nc)
        self.S = min(self.cap, self.vf*self.n)
        self.R = min(self.cap, (self.nj-self.n)*self.w)
        
    # inputs: autonomy level, array of headways [human, autonomous], vehicle length, road length
    # output: critical density, in vehicles per cell
    def critdens(self, autlevel):
        assert 0 <= autlevel <= 1
        return self.cell_length * self.num_active_lanes / (autlevel*self.headway[1] + (1.0-autlevel)*self.headway[0] + L)

    # maximum flow (vehicles per timestep)
    def maxflow(self, critdensity):
        return critdensity*self.vf
    
    def shockspeed(self, critdensity):
        if self.accidents.size > 0: # not the queue
            return critdensity*self.vf/(self.nj - critdensity)
        return np.inf

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class GeneralNetwork(gym.Env):
    def __init__(self, start_empty=False, demand=[1.993974,2.990961], init_learn_rate=0.5, constant_learn_rate=True, sim_duration=5.0, start_from_equilibrium=False, accident_param=0.6, demand_noise_std=[1.993974/100,2.990961/100]):
        self.sim_duration = sim_duration*secperhr
        self.start_empty = start_empty
        self.start_from_equilibrium = start_from_equilibrium and not start_empty
        self.init_learn_rate = init_learn_rate
        self.constant_learn_rate = constant_learn_rate
        self.demand = demand
        assert accident_param >= 0
        self.accident_param = accident_param
        self.demand_noise_std = demand_noise_std
        self.seed(0) # just in case we forget seeding
        self.initialize()
        assert np.isinf(self.cells[0].cap) and self.cells[0].cap > 0 # make sure cell 0 is the queue
    
    def set(self, property, value):
        if property.lower() == 'sim_duration':
            assert value >= 0
            self.sim_duration = value*secperhr
        elif property.lower() == 'start_empty':
            self.start_empty = value
        elif property.lower() == 'start_from_equilibrium':
            self.start_from_equilibrium = value
        elif property.lower() == 'init_learn_rate':
            assert value >= 0
            self.init_learn_rate = value
        elif property.lower() == 'constant_learn_rate':
            self.constant_learn_rate = value
        elif property.lower() == 'demand':
            assert np.min(value) >= 0
            self.demand = value
        elif property.lower() == 'accident_param':
            assert value >= 0
            self.accident_param = value
        elif property.lower() == 'demand_noise_std':
            assert np.min(value) >= 0
            self.demand_noise_std = value
        else:
            import warnings
            warnings.warn('Warning: No such property is defined. No action taken. List of properties: sim_duration, start_empty, start_from_equilibrium, p, init_learn_rate, constant_learn_rate, demand, accident_param')
        self.initialize()
        
    def get(self, property):
        if property.lower() == 'sim_duration':
            return self.sim_duration
        elif property.lower() == 'start_empty':
            return self.start_empty
        elif property.lower() == 'start_from_equilibrium':
            return self.start_from_equilibrium
        elif property.lower() == 'init_learn_rate':
            return self.init_learn_rate
        elif property.lower() == 'constant_learn_rate':
            return self.constant_learn_rate
        elif property.lower() == 'demand':
            return self.demand
        elif property.lower() == 'accident_param':
            return self.accident_param
        elif property.lower() == 'demand_noise_std':
            return self.demand_noise_std
        else:
            import warnings
            warnings.warn('Warning: No such property is defined. No output returned. List of properties: sim_duration, start_empty, start_from_equilibrium, p, init_learn_rate, constant_learn_rate, demand, accident_param')
    
    def downstream_of(self, cell_id):
        junc = np.array(self.junctions)
        return junc[np.argwhere([cell_id in x for x in junc[:,0]])[0,0]][1]
    def upstream_of(self, cell_id):
        junc = np.array(self.junctions)
        return junc[np.argwhere([cell_id in x for x in junc[:,1]])[0,0]][0]

    def initialize(self):
        self.max_step_size = int(self.sim_duration/T)
        
        # Create all cells independently
        self.cells = []
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, np.inf, 1)) # infinite capacity queue in the beginning
        self.cells.append(Cell(75*kmpermiles, 1.25*kmpermiles*1000*quantization, 2, 2))
        self.cells.append(Cell(75*kmpermiles, 1.25*kmpermiles*1000*quantization, 2, 2))
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 3, 3))
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 1, 1))
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 3, 3))
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 2, 2))
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 3, 3))
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 1, 1))
        self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 2, 2))
        self.num_cells = len(self.cells)

        # Connect the cells via junctions: [[a],[b,c]] means cell a feeds cells b and c
        self.junctions = []
        self.junctions.append([[],[0]]) # welcome to the network (almost), please wait in the queue first
        self.junctions.append([[0],[1,3]])
        self.junctions.append([[1,4],[2]])
        self.junctions.append([[2],[]]) # goodbye!
        self.junctions.append([[3],[4,5]])
        self.junctions.append([[5],[6,7]])
        self.junctions.append([[6],[8]])
        self.junctions.append([[7,8],[9]])
        self.junctions.append([[9],[]]) # goodbye!
        
        # Let's create an order of cells such that a downstream cell always comes before the upstream ones -- will be useful later
        cells_processed = [False]*len(self.cells)
        self.ordered_cell_ids = []
        while not np.all(cells_processed):
            unprocessed = np.argwhere(np.logical_not(cells_processed)).squeeze()
            for i in range(len(unprocessed)-1, -1, -1): # let's search backwards because cells will be probably added from upstream to downstream
                c_id = unprocessed[i]
                ds = self.downstream_of(c_id)
                if (not ds) or np.all([cells_processed[x] for x in ds]):
                    self.ordered_cell_ids.append(c_id)
                    cells_processed[c_id] = True
        
        # Having the cells and junctions, we now automatically generate the paths
        self.paths = [[0]]
        paths_complete = [False]
        while not np.all(paths_complete):
            curr_pathid = np.argwhere(np.logical_not(paths_complete))[0,0]
            curr_ds = self.downstream_of(self.paths[curr_pathid][-1])
            if curr_ds:
                for i in range(1,len(curr_ds)):
                    self.paths.append(self.paths[curr_pathid][:]) # [:] is necessary for making a deep copy
                    paths_complete.append(False)
                    self.paths[-1].append(curr_ds[i])
                self.paths[curr_pathid].append(curr_ds[0])
            else:
                paths_complete[curr_pathid] = True
        self.num_paths = len(self.paths)
            
            
        # expected number of accidents in one hour = self.accident_param
        self.p = self.accident_param*(self.sim_duration/secperhr) / (float(self.max_step_size) * np.sum([cell.num_lanes for cell in self.cells]))
        
        self.step_count = 0

        # start with uniform distribution
        self.human_distribution = np.array([1.0/self.num_paths]*self.num_paths)
        self.n_t_h = self.init_learn_rate # initial learning rate for humans
        self.aut_distribution = np.array([1.0/self.num_paths]*self.num_paths)
        self.n_t_a = self.init_learn_rate # initial learning rate for autonomous users (for selfish policy)

        self.T_init = self.max_step_size # if starting from an equilibrium, the number of time steps before beginning control

        demand = np.array(self.demand)*T
        self.demand_h = demand[0]
        self.demand_a = demand[1]

        # reward will be the difference between two value functions (of consecutive timesteps)
        self.last_total_num_of_cars = 0
        
    def get_demand(self):
        return max(self.demand_h + self.demand_noise_std[0]*self.np_random.randn(),0), max(self.demand_a + self.demand_noise_std[1]*self.np_random.randn(),0)
        #return self.demand_h, self.demand_a
        
    def reset(self):
        # reset preferences
        self.human_distribution = np.array([1.0/self.num_paths]*self.num_paths)
        self.n_t_h = self.init_learn_rate 
        self.aut_distribution = np.array([1.0/self.num_paths]*self.num_paths)
        self.n_t_a = self.init_learn_rate # initial learning rate for autonomous users (for selfish policy)
        
        for cell in self.cells:
            cell.reset(self.num_paths)

        if not self.start_empty:
            # then randomly set densities...
            for cell_id in range(1,self.num_cells): # do not randomize the queue
                self.cells[cell_id].seed(self.np_random.randint(1000000))
                downstream_path_ids = np.where([(cell_id in path) for path in self.paths])[0]
                self.cells[cell_id].randomize_state(downstream_path_ids)
            if self.start_from_equilibrium:
                self.go_to_equilibrium(self.T_init)
                
        self.step_count = 0
        self.last_total_num_of_cars = 0

        return self._get_obs()
        
    def _get_obs(self):
        # get cell and queue states
        cell_states = np.array([cell.state / cell.cell_length for cell in self.cells]).reshape(-1)
        if not np.isclose(self.accident_param, 0):
            lane_states = np.array([a for cell in self.cells for a in cell.accidents])
            lane_state = (lane_states > 0).astype(float) # the agent does not see how long it will take to clear out the accidents
            return np.concatenate((cell_states, lane_states))
        else:
            return cell_states
            
    def update_network(self, routing_matrix):
        rm = np.array(routing_matrix)
        assert rm.shape[0] == 2 # nh, na
        assert rm.shape[1] == self.num_paths
        assert np.all(np.isclose(rm.sum(axis=1), 1))
        
        # first, set mu and densities of the queue cell
        ds_temp = self.downstream_of(0)
        self.cells[0].state += self.get_demand()
        self.cells[0].mu = rm.T
        
        # second: there are no mistakes only happy accidents
        for cell in self.cells:
            cell.simulate_accident(self.p)
        
        # third, update all cells' R and S functions
        for cell in self.cells:
            cell.update_params()
        
        # fourth, calculate all f (flows between cells)
        self.f = np.zeros((self.num_cells,2))
        self.y = np.zeros((self.num_cells,2))
        for junction in self.junctions:
            if len(junction[0]) == 0:
                continue # queue is manually filled above
            elif len(junction[0]) == 1:
                parent_id = junction[0][0]
                parent = self.cells[parent_id]
                if len(junction[1]) == 0: # exiting the network
                    self.f[junction[0][0],:] += parent.S * np.array([1-parent.autonomy, parent.autonomy])
                elif len(junction[1]) == 1: # regular junction
                    child_id = junction[1][0]
                    child = self.cells[child_id]
                    self.f[parent_id,:] += min(parent.S, child.R) * np.array([1-parent.autonomy, parent.autonomy])
                    self.y[child_id,:] += self.f[parent_id,:]
                else: # diverge
                    beta = np.zeros((2,len(junction[1]))) # 2 for human and autonomous
                    for i in range(len(junction[1])):
                        child_id = junction[1][i]
                        child_in_paths = [(child_id in path) for path in self.paths]
                        beta[:,i] = parent.mu[child_in_paths,:].sum(axis=0)
                    assert np.all([np.isclose(b.sum(), 1) for b in beta])
                    if not np.isclose(parent.n, 0):
                        beta_overall = (beta[0,:]*parent.state[0] + beta[1,:]*parent.state[1]) / parent.n
                    else:
                        beta_overall = (beta[0,:] + beta[1,:]) / 2.
                    temp_f = min(parent.S, np.min([self.cells[junction[1][i]].R / beta_overall[i] for i in range(len(junction[1]))]))
                    self.f[parent_id,:] += temp_f * np.array([1-parent.autonomy, parent.autonomy])
                    for i in range(len(junction[1])):
                        child_id = junction[1][i]
                        self.y[child_id,:] += beta[:,i] * self.f[parent_id,:]
                    assert np.all(np.isclose(self.f[parent_id,:], self.y[junction[1],:].sum(axis=0)))

            else: # merge
                assert len(junction[1]) == 1
                child_id = junction[1][0]
                child = self.cells[child_id]
                parents_S = np.array([self.cells[c_id].S for c_id in junction[0]])
                parents_autonomy = np.array([self.cells[c_id].autonomy for c_id in junction[0]])
                parents_priority = np.array([self.cells[c_id].priority for c_id in junction[0]])
                parents_priority = parents_priority / parents_priority.sum()
                if parents_S.sum() <= child.R:
                    self.f[junction[0],:] += (parents_S * np.vstack((1-parents_autonomy, parents_autonomy))).T
                else:
                    temp = np.median([parents_S, child.R - parents_S.sum() + parents_S, parents_priority*child.R], axis=0)
                    self.f[junction[0],:] += (temp * np.vstack((1-parents_autonomy, parents_autonomy))).T
                self.y[child_id,:] += self.f[junction[0],:].sum(axis=0)
                
        # fifth, starting from the most downstream junction, update all mu and densities
        for c_id in self.ordered_cell_ids:
            if c_id > 0: # no update on mu if the cell is the queue -- its mu was updated in the first step
                U = self.upstream_of(c_id)
                mu_numerator = np.zeros((self.num_paths,2))
                for p in range(self.num_paths):
                    mu_numerator[p,:] = (self.f[U,:]*[self.cells[u].mu[p,:] for u in U]).sum(axis=0)
                    mu_numerator[p,:] += self.cells[c_id].mu[p,:]*(self.cells[c_id].state - self.f[c_id,:])
            
            # update the densities now
            self.cells[c_id].state += self.y[c_id,:]
            self.cells[c_id].state -= self.f[c_id,:]
            self.cells[c_id].state[np.isclose(self.cells[c_id].state,0)] = 0.
            
            # back to mu
            if c_id > 0:
                mu_denominator = self.cells[c_id].state
                downstream_path_ids = np.where([(c_id in path) for path in self.paths])[0]
                for type in range(2):
                    if np.isclose(mu_denominator[type], 0):
                        self.cells[c_id].mu[:,type] = 0.
                        self.cells[c_id].mu[downstream_path_ids,type] = 1. / len(downstream_path_ids)
                    else:
                        self.cells[c_id].mu[:,type] = mu_numerator[:,type] / mu_denominator[type]
                self.cells[c_id].mu = np.clip(self.cells[c_id].mu, 0., 1.) # because of numerical issues
            
    def go_to_equilibrium(self, T, no_accidents=True):
        for _ in range(T):
            self.aut_distribution, self.n_t_a = self.set_selfish_decision(self.aut_distribution, self.n_t_a)
            self.step(self.aut_distribution, no_accidents)
        
    # generic update function for either class of vehicle. pass in current distributions
    # and learning rate and it returns the updated distribution and learning rate
    def set_selfish_decision(self, dist, n_t):
        assert dist.size == self.num_paths
        assert n_t >= 0

        new_dist = dist*0
        latencies = self.measure_latencies()
        for p in range(self.num_paths):
            numerator = dist[p]*np.exp(-n_t*latencies[p])
            denominator = np.sum([dist[q]*np.exp(-n_t*latencies[q]) for q in range(self.num_paths)])
            new_dist[p] = numerator / denominator

        # update learning rate
        new_n_t = n_t
        if not self.constant_learn_rate:
            new_n_t = 1.0/(1.0/n_t + 1) # update learning rate from 1/t to 1/(t+1)
        
        return new_dist, new_n_t
        
    # measure latency that a person would face by entering path p.
    # calculate this by seeing how many time steps it takes to empty the currently existing cars in the path
    def measure_latencies(self):
        temp_self = copy.deepcopy(self)

        dummy_routing_matrix = self.np_random.rand(2,temp_self.num_paths)
        dummy_routing_matrix = dummy_routing_matrix / dummy_routing_matrix.sum(axis=1).reshape(-1,1)
        temp_self.update_network(dummy_routing_matrix) # so that f values will be calculated for cells
        temp_f = temp_self.f
        
        latencies = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            if np.isclose(self.cells[i].n, 0): # just in case some cells are empty and so f is 0
                latencies[i] = 0.
            elif np.isclose(temp_f[i].sum(), 0):
                latencies[i] = MANY_ITERS
            else:
                latencies[i] = self.cells[i].n / temp_f[i].sum()
        latencies = np.clip(latencies, [1./cell.vf for cell in self.cells], None) # cell latencies cannot be less than the free-flow latencies
        
        road_latencies = []
        for path in self.paths:
            road_latencies.append(latencies[path].sum())
        road_latencies = np.array(road_latencies)
        road_latencies -= latencies[0] # exclude the queue

        return road_latencies
        
    def step(self, action, no_accidents=False):
        num_h, num_a = self.get_demand()
        
        self.human_distribution, self.n_t_h = self.set_selfish_decision(self.human_distribution, self.n_t_h)
    
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.num_paths == 2:
            action = np.append(action[0], 1-action[0])
        else:
            if np.all(np.isclose(action,0)):
                action = np.array([1.]*self.num_paths)
            action = action / action.sum()
        # At this point, action is a vector whose sum is 1 and that consists of num_paths entries
        
        routing_matrix = [self.human_distribution, action]
        self.update_network(routing_matrix)
        
        total_num_of_cars = np.sum([cell.n for cell in self.cells])
        
        # take the difference with the previous time step as the reward
        reward = self.last_total_num_of_cars - total_num_of_cars
        self.last_total_num_of_cars = total_num_of_cars
        
        self.step_count += 1
        return self._get_obs(), reward, self.step_count>=self.max_step_size, {}
        
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    @property
    def action_space(self):
        if self.num_paths == 2:
            return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        return spaces.Box(low=0, high=1, shape=(self.num_paths,), dtype=np.float32)
    
    @property
    def observation_space(self):
        # observation should include densities of all cells, as well as the number of human-driven and autonomous
        # vehicles in the queue. Maybe TODO: also include the autonomous vehicle demand
        dim_cells = 2*self.num_cells
        if not np.isclose(self.accident_param,0):
            dim_lanes = np.sum([cell.num_lanes for cell in self.cells[1:]]) # exclude the queue for accidents
            low = np.concatenate((np.zeros((dim_cells,)), np.zeros((dim_lanes,))))
            high = np.concatenate((np.concatenate([[cell.nj/cell.cell_length]*2 for cell in self.cells]).reshape(-1), np.ones((dim_lanes,))*np.inf))
        else:
            low = np.zeros((dim_cells,))
            high = np.concatenate([[cell.nj/cell.cell_length]*2 for cell in self.cells]).reshape(-1)
        return spaces.Box(low=low, high=high, dtype=np.float32)