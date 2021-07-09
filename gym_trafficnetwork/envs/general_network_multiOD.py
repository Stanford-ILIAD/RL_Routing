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
        self.randomize_mu(downstream_path_ids)
        aut = self.np_random.rand()
        n = self.np_random.rand() * 1.2 * self.critdens(aut)
        self.state = np.array([(1-aut) * n, aut * n])
        self.update_params()
        
    def randomize_mu(self, downstream_path_ids):
        self.mu[downstream_path_ids,:] = self.np_random.rand(len(downstream_path_ids),2)
        self.mu /= self.mu.sum(axis=0)
    
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

class GeneralNetworkMultiOD(gym.Env):
    def __init__(self, start_empty=False, demand=[[[1.993974,2.990961],[1.993974,2.990961]],[[1.993974,2.990961],[1.993974,2.990961]]], init_learn_rate=0.5, constant_learn_rate=True, sim_duration=5.0, start_from_equilibrium=False, accident_param=0.6, demand_noise_std=[[[1.993974/100,2.990961/100],[1.993974/100,2.990961/100]],[[1.993974/100,2.990961/100],[1.993974/100,2.990961/100]]]):
        self.sim_duration = sim_duration*secperhr
        self.start_empty = start_empty
        self.start_from_equilibrium = start_from_equilibrium and not start_empty
        self.init_learn_rate = init_learn_rate
        self.constant_learn_rate = constant_learn_rate
        self.num_origins = len(demand)
        self.num_destinations = len(demand[0])
        self.num_ods = self.num_origins * self.num_destinations
        self.queue_ids = np.arange(0, self.num_origins)
        self.orig_demand = np.array(demand)
        assert accident_param >= 0
        self.accident_param = accident_param
        self.demand_noise_std = np.array(demand_noise_std)
        self.seed(0) # just in case we forget seeding
        self.initialize()
        for i in self.queue_ids:
            assert np.isinf(self.cells[i].cap) and self.cells[i].cap > 0 # make sure cell i is a queue
    
    def set(self, property, value):
        if property.lower() == 'sim_duration':
            assert value >= 0
            self.sim_duration = value*secperhr
            self.max_step_size = int(self.sim_duration/T)
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
            self.orig_demand = np.array(value)
        elif property.lower() == 'accident_param':
            assert value >= 0
            self.accident_param = value
        elif property.lower() == 'demand_noise_std':
            assert np.min(value) >= 0
            self.demand_noise_std = value
        else:
            import warnings
            warnings.warn('Warning: No such property is defined. No action taken. List of properties: sim_duration, start_empty, start_from_equilibrium, p, init_learn_rate, constant_learn_rate, demand, accident_param')
        #self.initialize()
        
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
            return self.orig_demand
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
        for _ in range(self.num_origins):
            self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, np.inf, 1)) # infinite capacity queue in the beginning
        '''
        # Debug Network
        for _ in range(self.num_origins, 10):
            self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 2, 1))
        '''
        #'''
        # OW Network
        for _ in range(self.num_origins, 106):
            self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 2, 1))
        #'''
        '''
        # Simplified OW Network
        for _ in range(self.num_origins, 74):
            self.cells.append(Cell(60*kmpermiles, 1*kmpermiles*1000*quantization, 2, 1))
        '''
        self.num_cells = len(self.cells)

        # Connect the cells via junctions: [[a],[b,c]] means cell a feeds cells b and c
        self.junctions = []
        self.junctions.append([[],[0]]) # welcome to the network (almost), please wait in the queue first
        self.junctions.append([[],[1]]) # welcome to the network (almost), please wait in the queue first
        
        '''
        # Debug Network
        self.junctions.append([[0,2],[3,4,7]])
        self.junctions.append([[1,3],[2,5,6]])
        self.junctions.append([[4,6,9],[8,10]]) # goodbye!
        self.junctions.append([[5,7,8],[9,11]]) # goodbye!
        '''
        #'''
        # OW Network
        self.junctions.append([[0,4,6,8],[2,9,7]])
        self.junctions.append([[2,5,10],[3,4,11]])
        self.junctions.append([[1,3,12],[5,13]])
        self.junctions.append([[7,14,16,18],[6,15,17,19]])
        self.junctions.append([[9,20],[8,21]])
        self.junctions.append([[11,22],[10,23]])
        self.junctions.append([[13,24],[12,25]])
        self.junctions.append([[15,26],[14,27]])
        self.junctions.append([[17,28],[16,29]])
        self.junctions.append([[19,30],[18,31]])
        self.junctions.append([[21,32],[20,33]])
        self.junctions.append([[23,34],[22,35]])
        self.junctions.append([[25,36],[24,37]])
        self.junctions.append([[27,44],[26,45]])
        self.junctions.append([[31,33,35,39,46,50],[30,32,34,38,47,51]])
        self.junctions.append([[38,41],[39,40]])
        self.junctions.append([[40,43],[41,42]])
        self.junctions.append([[37,42,54],[36,43,55]])
        self.junctions.append([[47,48],[46,49]])
        self.junctions.append([[51,52],[50,53]])
        self.junctions.append([[55,56],[54,57]])
        self.junctions.append([[45,59,66],[44,58,67]])
        self.junctions.append([[58,61],[59,60]])
        self.junctions.append([[29,49,60,63,72,74],[28,48,61,62,73,75]])
        self.junctions.append([[62,65],[63,64]])
        self.junctions.append([[53,57,64,78],[52,56,65,79]])
        self.junctions.append([[67,68],[66,69]])
        self.junctions.append([[69,70],[68,71]])
        self.junctions.append([[75,76],[74,77]])
        self.junctions.append([[77,80],[76,81]])
        self.junctions.append([[71,83,90],[70,82,91]])
        self.junctions.append([[82,85],[83,84]])
        self.junctions.append([[73,84,87,96,98],[72,85,86,97,99]])
        self.junctions.append([[86,89],[87,88]])
        self.junctions.append([[79,81,88,104],[78,80,89,105]])
        self.junctions.append([[91,93],[90,92,106]]) # goodbye!
        self.junctions.append([[92,95],[93,94]])
        self.junctions.append([[94,97],[95,96]])
        self.junctions.append([[99,100],[98,101]])
        self.junctions.append([[101,102],[100,103]])
        self.junctions.append([[103,105],[102,104,107]]) # goodbye!
        #'''
        '''
        # Simplified OW Network
        self.junctions.append([[0,3,6,8],[2,7,9]])
        self.junctions.append([[2,5],[3,4]])
        self.junctions.append([[1,4,14,20],[5,15,21]])
        self.junctions.append([[7,27,34],[6,26,35]])
        self.junctions.append([[9,10],[8,11]])
        self.junctions.append([[11,12],[10,13]])
        self.junctions.append([[26,29],[27,28]])
        self.junctions.append([[13,19,28,31,42,40],[12,18,29,30,43,41]])
        self.junctions.append([[15,16],[14,17]])
        self.junctions.append([[17,18],[16,19]])
        self.junctions.append([[30,33],[31,32]])
        self.junctions.append([[21,22],[20,23]])
        self.junctions.append([[23,24],[22,25]])
        self.junctions.append([[25,32,48],[24,33,49]])
        self.junctions.append([[43,44],[42,45]])
        self.junctions.append([[45,46],[44,47]])
        self.junctions.append([[47,49,56,72],[46,48,57,73]])
        self.junctions.append([[54,57],[55,56]])
        self.junctions.append([[41,52,55,64,66],[40,53,54,65,67]])
        self.junctions.append([[50,53],[51,52]])
        self.junctions.append([[39,51,58],[38,50,59]])
        self.junctions.append([[35,36],[34,37]])
        self.junctions.append([[37,38],[36,39]])
        self.junctions.append([[59,61],[58,60,74]]) # goodbye!
        self.junctions.append([[60,63],[61,62]])
        self.junctions.append([[62,65],[63,64]])
        self.junctions.append([[67,68],[66,69]])
        self.junctions.append([[69,70],[68,71]])
        self.junctions.append([[71,73],[70,72,75]]) # goodbye!
        '''

              
        # Having the cells and junctions, we now automatically generate the paths
        self.paths = [[x] for x in range(self.num_origins)]
        paths_complete = [False] * self.num_origins
        while not np.all(paths_complete):
            curr_pathid = np.argwhere(np.logical_not(paths_complete))[0,0]
            curr_ds = self.downstream_of(self.paths[curr_pathid][-1])
            if np.any([x in curr_ds for x in self.paths[curr_pathid]]): # this block removes the cyclic paths
                del self.paths[curr_pathid]
                del paths_complete[curr_pathid]
                continue
            for i in range(1,len(curr_ds)):
                self.paths.append(self.paths[curr_pathid][:]) # [:] is necessary for making a deep copy
                self.paths[-1].append(curr_ds[i])
                paths_complete.append(curr_ds[i] >= self.num_cells)
            self.paths[curr_pathid].append(curr_ds[0])
            paths_complete[curr_pathid] = curr_ds[0] >= self.num_cells

        # remove the cells that are not part of any path
        c_id = self.num_cells - 1
        while c_id >= self.num_origins:
            if not np.any([(c_id in p) for p in self.paths]):
                del self.cells[c_id]
                for j_id in range(len(self.junctions)):
                    junction = self.junctions[j_id]
                    if c_id in junction[0]:
                        junction[0].remove(c_id)
                    elif c_id in junction[1]:
                        junction[1].remove(c_id)
                    for i in range(len(junction[0])):
                        if junction[0][i] > c_id:
                            junction[0][i] = junction[0][i] - 1
                    for i in range(len(junction[1])):
                        if junction[1][i] > c_id:
                            junction[1][i] = junction[1][i] - 1
                    if (not junction[0] and not junction[1][0] < self.num_origins) or not junction[1]:
                        del self.junctions[j_id]
            c_id -= 1
        self.num_cells = len(self.cells)
                    
        # re-compute the paths
        self.paths = [[x] for x in range(self.num_origins)]
        paths_complete = [False] * self.num_origins
        while not np.all(paths_complete):
            curr_pathid = np.argwhere(np.logical_not(paths_complete))[0,0]
            curr_ds = self.downstream_of(self.paths[curr_pathid][-1])
            if np.any([x in curr_ds for x in self.paths[curr_pathid]]): # this block removes the cyclic paths
                del self.paths[curr_pathid]
                del paths_complete[curr_pathid]
                continue
            for i in range(1,len(curr_ds)):
                self.paths.append(self.paths[curr_pathid][:]) # [:] is necessary for making a deep copy
                self.paths[-1].append(curr_ds[i])
                paths_complete.append(curr_ds[i] >= self.num_cells)
            self.paths[curr_pathid].append(curr_ds[0])
            paths_complete[curr_pathid] = curr_ds[0] >= self.num_cells
        self.num_paths = len(self.paths)
        self.paths_od = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        self.path_ids_od = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        self.num_paths_od = np.zeros((self.num_origins, self.num_destinations), dtype=int)
        for o_id in range(self.num_origins):
            for d_id in range(self.num_destinations):
                self.path_ids_od[o_id][d_id] = [i for i in range(self.num_paths) if self.paths[i][0]==o_id and self.paths[i][-1] == self.num_cells + d_id]
                self.paths_od[o_id][d_id] = [path for path in self.paths if path[0]==o_id and path[-1] == self.num_cells + d_id]
                self.num_paths_od[o_id,d_id] = len(self.paths_od[o_id][d_id])

        # expected number of accidents in one hour = self.accident_param
        self.p = self.accident_param*(self.sim_duration/secperhr) / (float(self.max_step_size) * np.sum([cell.num_lanes for cell in self.cells]))
        
        self.step_count = 0

        # start with uniform distribution
        self.human_distribution = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        self.aut_distribution = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        for o_id in range(self.num_origins):
            for d_id in range(self.num_destinations):
                self.human_distribution[o_id][d_id] = np.array([1.0/self.num_paths_od[o_id,d_id]]*self.num_paths_od[o_id,d_id])
                self.aut_distribution[o_id][d_id] = np.array([1.0/self.num_paths_od[o_id,d_id]]*self.num_paths_od[o_id,d_id])
        self.n_t_h = self.init_learn_rate # initial learning rate for humans
        self.n_t_a = self.init_learn_rate # initial learning rate for autonomous users (for selfish policy)

        self.T_init = self.max_step_size # if starting from an equilibrium, the number of time steps before beginning control

        # reward will be the difference between two value functions (of consecutive timesteps)
        self.last_total_num_of_cars = 0
        
    def get_demand(self):
        return np.maximum(self.demand + self.demand_noise_std*self.np_random.randn(*self.demand.shape)*np.sqrt(T),0)
        
    def reset(self):
        self.demand = self.orig_demand*T
    
        # reset preferences
        self.human_distribution = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        self.aut_distribution = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        for o_id in range(self.num_origins):
            for d_id in range(self.num_destinations):
                self.human_distribution[o_id][d_id] = np.array([1.0/self.num_paths_od[o_id,d_id]]*self.num_paths_od[o_id,d_id])
                self.aut_distribution[o_id][d_id] = np.array([1.0/self.num_paths_od[o_id,d_id]]*self.num_paths_od[o_id,d_id])
        self.n_t_h = self.init_learn_rate # initial learning rate for humans
        self.n_t_a = self.init_learn_rate # initial learning rate for autonomous users (for selfish policy)
        
        for cell in self.cells:
            cell.reset(self.num_paths)

        if not self.start_empty:
            # then randomly set densities...
            for cell_id in range(self.num_origins, self.num_cells): # do not randomize the queues
                self.cells[cell_id].seed(self.np_random.randint(1000000))
                downstream_path_ids = np.where([(cell_id in path) for path in self.paths])[0]
                self.cells[cell_id].randomize_state(downstream_path_ids)
            if self.start_from_equilibrium:
                self.go_to_equilibrium(self.T_init)
        else:
            for cell_id in range(self.num_origins, self.num_cells): # do not randomize the queues
                self.cells[cell_id].seed(self.np_random.randint(1000000))
                downstream_path_ids = np.where([(cell_id in path) for path in self.paths])[0]
                self.cells[cell_id].randomize_mu(downstream_path_ids)
                
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
            
    def update_network(self, routing_data):
        human_routing, aut_routing = routing_data
        assert len(human_routing) == len(aut_routing) == self.num_origins
        assert len(human_routing[0]) == len(aut_routing[0]) == self.num_destinations
        for o_id in range(self.num_origins):
            for d_id in range(self.num_destinations):
                assert len(human_routing[o_id][d_id]) == len(aut_routing[o_id][d_id]) == self.num_paths_od[o_id,d_id]
                assert np.isclose(human_routing[o_id][d_id].sum(), 1), str(human_routing[o_id][d_id].sum())
                assert np.isclose(aut_routing[o_id][d_id].sum(), 1), str(aut_routing[o_id][d_id].sum())
        
        # first, set mu and densities of the queue cell
        demand = self.get_demand()
        routing_matrix = np.zeros((self.num_origins,self.num_paths,2))
        for o_id in range(self.num_origins):
            for d_id in range(self.num_destinations):
                routing_matrix[o_id,self.path_ids_od[o_id][d_id],0] = demand[o_id,d_id,0] * human_routing[o_id][d_id]
                routing_matrix[o_id,self.path_ids_od[o_id][d_id],1] = demand[o_id,d_id,1] * aut_routing[o_id][d_id]
        routing_matrix /= routing_matrix.sum(axis=1).reshape((self.num_origins,1,2))
        for o_id in range(self.num_origins):
            self.cells[o_id].state += demand[o_id].sum(axis=0)
            self.cells[o_id].mu = routing_matrix[o_id]

        # second: there are no mistakes, only happy accidents
        for cell in self.cells:
            cell.simulate_accident(self.p)
        
        # third, update all cells' R and S functions
        for cell in self.cells:
            cell.update_params()
               
        # fourth, calculate all f (flows between cells) and prepare mu's
        self.f = np.zeros((self.num_cells,2))
        self.y = np.zeros((self.num_cells,2))
        big_y = [[] for _ in range(len(self.junctions))]
        mu_numerator = np.zeros((self.num_cells, self.num_paths, 2))
        for j_id in range(len(self.junctions)):
            junction = self.junctions[j_id]
            if len(junction[0]) == 0:
                continue # queue is manually filled above
            
            beta = np.zeros((len(junction[0]),len(junction[1]),2)) # 2 for human and autonomous
            beta_overall = np.zeros((len(junction[0]),len(junction[1])))
            increase_rate = np.zeros((len(junction[0]),len(junction[1])))
            for i in range(len(junction[0])):
                parent = self.cells[junction[0][i]]
                for j in range(len(junction[1])):
                    child_id = junction[1][j]
                    child_in_paths = [(child_id in path) for path in self.paths]
                    beta[i,j,:] = parent.mu[child_in_paths,:].sum(axis=0)
                if not np.isclose(parent.n, 0):
                    beta_overall[i] = (beta[i,:,0]*parent.state[0] + beta[i,:,1]*parent.state[1]) / parent.n
                else:
                    beta_overall[i] = (beta[i,:,0] + beta[i,:,1]) / 2.
                increase_rate[i] = parent.S * beta_overall[i]
            if not np.all([np.isclose(beta[i,:,type].sum(), 1) for type in range(2) for i in range(len(junction[0]))]):
                import pdb; pdb.set_trace()
            #for beta_type in beta:
            #    for b in beta_type:
            #         b = b / b.sum()
                       
            y = np.zeros((len(junction[0]),len(junction[1])))
            big_y[j_id] = np.zeros((len(junction[0]), len(junction[1]), 2)) # 2 for human and autonomous
            active_pairs = np.full((len(junction[0]), len(junction[1])), True, dtype=bool)
            active_pairs[np.isclose(increase_rate,0)] = False
            while np.any(active_pairs):
                ir_active = increase_rate[active_pairs]
                y_active = y[active_pairs]
                min_coeff = (ir_active[0] - y_active[0]) / ir_active[0] # the division is the same for all indices, so we use 0
                bottleneck_j = -1
                for j in range(len(junction[1])):
                    if not np.any(active_pairs[:,j]) or junction[1][j] >= self.num_cells: # if not active or if an exit cell
                        continue
                    coeff = (self.cells[junction[1][j]].R - y[:,j].sum()) / increase_rate[:,j].sum()
                    if coeff < min_coeff:
                        min_coeff = coeff
                        bottleneck_j = j
                old_y = y.copy()
                y[active_pairs] += ir_active * min_coeff
                new_flow = y - old_y
                for i in range(len(junction[0])):
                    autonomy = self.cells[junction[0][i]].autonomy
                    for j in range(len(junction[1])):
                        if active_pairs[i,j]:
                            mu = self.cells[junction[0][i]].mu.copy()
                            ij_aut = autonomy * beta[i,j,1] / ((1-autonomy) * beta[i,j,0] + autonomy * beta[i,j,1])
                            big_y[j_id][i,j,:] += [new_flow[i,j] * (1-ij_aut), new_flow[i,j] * ij_aut]
                            downstream_path_flags = np.array([(junction[1][j] in path) for path in self.paths])
                            mu[np.logical_not(downstream_path_flags),:] = 0.
                            for type in range(2):
                                if np.isclose(mu[:,type].sum(),0):
                                    mu[downstream_path_flags,type] = 1. / np.sum(downstream_path_flags)
                                else:
                                    mu[downstream_path_flags,type] /= mu[downstream_path_flags,type].sum()
                            
                            if junction[1][j] < self.num_cells: # if not an exit cell
                                mu_numerator[junction[1][j],:,0] += new_flow[i,j] * (1-ij_aut) * mu[:,0]
                                mu_numerator[junction[1][j],:,1] += new_flow[i,j] * ij_aut * mu[:,1]
                            if junction[0][i] >= self.num_origins: # if not a queue
                                mu_numerator[junction[0][i],:,0] -= new_flow[i,j] * (1-ij_aut) * mu[:,0]
                                mu_numerator[junction[0][i],:,1] -= new_flow[i,j] * ij_aut * mu[:,1]
                if bottleneck_j == -1:
                    active_pairs *= False # all of them become inactive
                else:
                    active_pairs[:,bottleneck_j] = False
            
            temp_f = np.zeros((len(junction[0]), ))
            for i in range(len(junction[0])):
                temp_f[i] = y[i,:].sum()
                
            for j in range(len(junction[1])):
                if not junction[1][j] >= self.num_cells: # if not an exit cell
                    self.y[junction[1][j],:] = big_y[j_id][:,j,:].sum(axis=0)

            for i in range(len(junction[0])):
                self.f[junction[0][i],:] = big_y[j_id][i].sum(axis=0)
                
            if np.all(np.array(junction[1]) < self.num_cells):
                assert np.isclose(np.sum([self.f[i,0] for i in junction[0]]), np.sum([self.y[i,0] for i in junction[1]]))
                assert np.isclose(np.sum([self.f[i,1] for i in junction[0]]), np.sum([self.y[i,1] for i in junction[1]]))
            
            
        # sixth, update all mu and densities
        for c_id in range(self.num_cells):
            mu_numerator[c_id] += self.cells[c_id].mu * self.cells[c_id].state
            mu_numerator[c_id] = np.clip(mu_numerator[c_id], 0., None) # because of numerical issues
        # update the densities now
        for c_id in range(self.num_cells):
            self.cells[c_id].state = self.cells[c_id].state + self.y[c_id,:] - self.f[c_id,:]
            self.cells[c_id].state[self.cells[c_id].state < 0] = 0. # numerical issues
            #if c_id >= self.num_origins:
            #    assert np.all(np.isclose(mu_numerator[c_id].sum(axis=0), self.cells[c_id].state))
        # update mu's
        for c_id in range(self.num_cells):
            if c_id >= self.num_origins: # not a queue
                mu_denominator = mu_numerator[c_id].sum(axis=0) # or self.cells[c_id].state, but might raise numerical issues
                downstream_path_ids = np.where([(c_id in path) for path in self.paths])[0]
                for type in range(2):
                    if np.isclose(mu_denominator[type], 0):
                        self.cells[c_id].mu[:,type] = 0.
                        self.cells[c_id].mu[downstream_path_ids,type] = 1. / len(downstream_path_ids)
                    else:
                        self.cells[c_id].mu[:,type] = mu_numerator[c_id,:,type] / mu_denominator[type]
                self.cells[c_id].mu = np.clip(self.cells[c_id].mu, 0., 1.) # because of numerical issues
        
        #import pdb; pdb.set_trace()
        
    def test(self):
         print([self.cells[i].mu[[(i in path) for path in self.paths]].sum() for i in range(self.num_cells)] )
        
    def go_to_equilibrium(self, T, no_accidents=True):
        for _ in range(T):
            self.aut_distribution, self.n_t_a = self.set_selfish_decision(self.aut_distribution, self.n_t_a)
            self.step(self.aut_distribution, no_accidents)
        
    # generic update function for either class of vehicle. pass in current distributions
    # and learning rate and it returns the updated distribution and learning rate
    def set_selfish_decision(self, dist, n_t):
        assert len(dist) == self.num_origins and len(dist[0]) == self.num_destinations
        for i in range(self.num_origins):
            for j in range(self.num_destinations):
                assert len(dist[i][j]) == self.num_paths_od[i,j]
        assert n_t >= 0

        new_dist = copy.deepcopy(dist)
        for i in range(self.num_origins):
            for j in range(self.num_destinations):
                new_dist[i][j] = dist[i][j]*0

        latencies = self.measure_latencies()
        for i in range(self.num_origins):
            for j in range(self.num_destinations):
                numerator = np.zeros((self.num_paths_od[i,j],))
                for p in range(self.num_paths_od[i,j]):
                    assert latencies[i][j][p] >= 0, str(latencies[i][j][p])
                    numerator[p] = dist[i][j][p]*np.exp(-n_t*latencies[i][j][p])
                new_dist[i][j] = numerator / numerator.sum()

        # update learning rate
        new_n_t = n_t
        if not self.constant_learn_rate:
            new_n_t = 1.0/(1.0/n_t + 1) # update learning rate from 1/t to 1/(t+1)
        
        return new_dist, new_n_t
        
    # measure latency that a person would face by entering path p.
    # calculate this by seeing how many time steps it takes to empty the currently existing cars in the path
    # assuming that the current flows will be maintained
    def measure_latencies(self):
        temp_self = copy.deepcopy(self)
        temp_self2 = copy.deepcopy(self)

        dummy_routing_data_hum = [[self.np_random.rand(self.num_paths_od[i,j]) for j in range(self.num_destinations)] for i in range(self.num_origins)]
        for o_id in range(self.num_origins):
            for d_id in range(self.num_destinations):
                dummy_routing_data_hum[o_id][d_id] = dummy_routing_data_hum[o_id][d_id] / dummy_routing_data_hum[o_id][d_id].sum()
        dummy_routing_data_aut = [[self.np_random.rand(self.num_paths_od[i,j]) for j in range(self.num_destinations)] for i in range(self.num_origins)]
        for o_id in range(self.num_origins):
            for d_id in range(self.num_destinations):
                dummy_routing_data_aut[o_id][d_id] = dummy_routing_data_aut[o_id][d_id] / dummy_routing_data_aut[o_id][d_id].sum()
        dummy_routing_data = [dummy_routing_data_hum, dummy_routing_data_aut]
        
        temp_self.update_network(dummy_routing_data) # so that f values will be calculated for cells
        temp_f = temp_self.f
        
        latencies = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            if np.isclose(self.cells[i].n, 0): # just in case some cells are empty and so f is 0
                latencies[i] = 1./self.cells[i].vf # free-flow latency
            elif np.isclose(temp_f[i].sum(), 0):
                latencies[i] = MANY_ITERS
            else:
                latencies[i] = self.cells[i].n / temp_f[i].sum()
                if temp_f[i].sum() < 0:
                    print(np.argmin(temp_f[:,0]))
                    print(np.argmin(temp_f[:,1]))
                    print(temp_f[np.argmin(temp_f[:,0])])
                    print(temp_f[np.argmin(temp_f[:,1])])
                    temp_self2.update_network(dummy_routing_data, debug=True)

        #latencies = np.clip(latencies, [1./cell.vf for cell in self.cells], None) # cell latencies cannot be less than the free-flow latencies
        
        road_latencies = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        for i in range(self.num_origins):
            for j in range(self.num_destinations):
                for path in self.paths_od[i][j]:
                    road_latencies[i][j].append(latencies[path[:-1]].sum()) # exclude the last cell of the path because it is an exit cell
                road_latencies[i][j] = np.array(road_latencies[i][j]) - latencies[i] # exclude the queue

        return road_latencies
        
    def step(self, shrunk_action, no_accidents=False):
        demand = self.get_demand()
        num_h = demand[:,0]
        num_a = demand[:,1]
        
        self.human_distribution, self.n_t_h = self.set_selfish_decision(self.human_distribution, self.n_t_h)
    
        action = np.zeros(self.num_paths, dtype=float)
        action[[51,15,0,305,135,145,282,50,154,321,146,144,143,142,137,141,156,5,481,482,267,131,13,275,457,97,95,703,266,435,9,277,437,441,442,443,444,446,722,721]] = shrunk_action
    
        action = np.clip(action, 0., 1.)
        action_reshaped = [[[] for x in range(self.num_destinations)] for y in range(self.num_origins)]
        for i in range(self.num_origins):
            for j in range(self.num_destinations):
                action_reshaped[i][j] = action[:self.num_paths_od[i,j]]
                action = action[self.num_paths_od[i,j]:]
                if np.all(np.isclose(action_reshaped[i][j],0)):
                    action_reshaped[i][j] = np.array([1.]*self.num_paths_od[i,j])
                action_reshaped[i][j] = action_reshaped[i][j] / action_reshaped[i][j].sum()
        # At this point, action is a num_origins x num_destinations list where each entry is a vector whose sum is 1
        
        routing_data = [self.human_distribution, action_reshaped]
        self.update_network(routing_data)
        
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
        return spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32)
    
    @property
    def observation_space(self):
        # observation should include densities of all cells, as well as the number of human-driven and autonomous
        # vehicles in the queue. Maybe TODO: also include the autonomous vehicle demand
        dim_cells = 2*self.num_cells
        if not np.isclose(self.accident_param,0):
            dim_lanes = np.sum([cell.num_lanes for cell in self.cells[self.num_origins:]]) # exclude the queues for accidents
            low = np.concatenate((np.zeros((dim_cells,)), np.zeros((dim_lanes,))))
            high = np.concatenate((np.concatenate([[cell.nj/cell.cell_length]*2 for cell in self.cells]).reshape(-1), np.ones((dim_lanes,))*np.inf))
        else:
            low = np.zeros((dim_cells,))
            high = np.concatenate([[cell.nj/cell.cell_length]*2 for cell in self.cells]).reshape(-1)
        return spaces.Box(low=low, high=high, dtype=np.float32)