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
	def __init__(self, vfkph, cell_length, num_lanes):
		self.vfms = vfkph*1000.0/secperhr # free-flow speed in meters per second
		self.headway = [self.vfms * tauh, self.vfms * taua] # meters
		self.cell_length = cell_length # meters
		self.accidents = np.array([0]*num_lanes) # nonpositive values will mean the lane is clear. Positive values will be the number of timesteps required to clear.
		self.vf = T * self.vfms / self.cell_length # free-flow speed in cells per time step
		self.nj = self.num_active_lanes * self.cell_length / L # jam density (vehicles per cell)

		self.reset()

	@property
	def num_active_lanes(self):
		return np.sum(self.accidents <= MACHINE_PRECISION)
		
	@property
	def num_lanes(self):
		return len(self.accidents)
		
	def reset(self):
		self.state = [0]*4
		self.accidents *= 0 # clear all accidents
		self.update_params()
	
	# creates an accident with probability p and makes sure the road is not completely blocked
	def simulate_accident(self, p=5e-6):
		self.accidents -= 1
		if self.state[2] > 1 + MACHINE_PRECISION: # no accidents if there are fewer than 2 cars
			for i in range(self.num_lanes):
				if self.np_random.rand() < p and not (self.accidents[i] <= 0 and self.num_active_lanes == 1): # makes sure we are not entirely blocking the road
					# Based on https://www.denenapoints.com/long-usually-take-clear-accident-scene-houston/
					# The average time to clear an accident was about 33-34 minutes in 2011 in Houston. Let's say it is now 30 minutes = 0.5 hours.
					self.accidents[i] += self.np_random.poisson(lam=0.5*secperhr/T) # so that we allow for a second accident in the same lane
		self.accidents = np.clip(self.accidents, 0, None) # let's not make it go to negative for simpler observation model for RL
	
	# calculate the relevant parameters for the cell (depends on autonomy level)
	def update_params(self):
		self.nc = self.critdens(self.state[3])
		self.w = self.shockspeed(self.nc)
		self.cap = self.maxflow(self.nc)
		
	# inputs: autonomy level, array of headways [human, autonomous], vehicle length, road length
	# output: critical density, in vehicles per cell
	def critdens(self, autlevel):
		assert(autlevel >= 0)
		assert(autlevel <= 1)
		return self.cell_length * self.num_active_lanes / (autlevel*self.headway[1] + (1.0-autlevel)*self.headway[0] + L)

	# maximum flow (vehicles per timestep)
	def maxflow(self, critdensity):
		return critdensity*self.vf
	
	def shockspeed(self, critdensity):
		return critdensity*self.vf/(self.nj - critdensity)
		
	def step(self, hum_in, aut_in, previous_cell=None, next_cell=None):
		# update_params() will be executed by the Road class for each cell before running the step method
		# accidents will also be handled by the Road class so that parameters will be properly updated
	
		self.fout = [0]*3
		if next_cell is None:
			self.fout[2] = np.amin( [ self.vf*self.state[2], self.cap ] )
		else:
			self.fout[2] = np.amin( [ self.vf*self.state[2], next_cell.w*(next_cell.nj-next_cell.state[2]), self.cap ] )
		self.fout[0] = (1-self.state[3])*self.fout[2]
		self.fout[1] = self.state[3]*self.fout[2]
		
		# do the updates
		self.state[0] = self.state[0] - self.fout[0] + hum_in
		self.state[1] = self.state[1] - self.fout[1] + aut_in
		self.state[2] = self.state[0] + self.state[1]
		if self.state[2] > MACHINE_PRECISION:
			self.state[3] = self.state[1] / self.state[2]
			self.state[3] = np.clip(self.state[3], 0, 1) # numerical issues cause -1e-16 like things...
		else:
			self.state[3] = 0
			
		return self.fout

	def seed(self, seed):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

		
class Road:
	def __init__(self, cells, probability_of_accident):
		self.cells = cells # a list of Cell objects
		self.num_cells = len(self.cells)
		self.probability_of_accident = probability_of_accident
		
		self.reset()
		
	# arguments: human and autonomous incoming flows (number of vehicles),
	# Note that road length should be equal to vf in our formulation
	def step(self, hum_in, aut_in, no_accidents=False):

		# update critical density and the other relevant parameters for each cell
		for i in range(self.num_cells):
			if not no_accidents:
				self.cells[i].simulate_accident(self.probability_of_accident)
			self.cells[i].update_params()
		
		# update cells
		for i in range(self.num_cells):
			if i == 0:
				if self.num_cells == 1:
					hum_in, aut_in, _ = self.cells[i].step(hum_in, aut_in, None, None)
				else:
					hum_in, aut_in, _ = self.cells[i].step(hum_in, aut_in, None, self.cells[i+1])
			elif i < self.num_cells - 1:
				hum_in, aut_in, _ = self.cells[i].step(hum_in, aut_in, self.cells[i-1], self.cells[i+1])
			else:
				hum_in, aut_in, _ = self.cells[i].step(hum_in, aut_in, self.cells[i-1], None)
		
		# update again, so incoming flow does not exceed capacity
		for i in range(self.num_cells):
			self.cells[i].update_params()

		return self.value()

	# calculate the flow entering the first cell
	def max_incoming(self):
		return np.amin( [ self.cells[0].w*(self.cells[0].nj - self.cells[0].state[2]), self.cells[0].cap ] )
		
	# measure latency that a person would face by entering the current road.
	# calculate this by seeing how many time steps it takes to empty the road
	# to do this we will step forward long enough to empty the road, then restore the original state
	def measure_latency(self):

		temp_cells = copy.deepcopy(self.cells)

		empty = (np.sum([cell.state[2] for cell in self.cells]) < MACHINE_PRECISION)
		latency = 0

		while not empty:
			self.step(0, 0, no_accidents=True)
			latency += 1
			if latency >= MANY_ITERS:
				# assert(True==False) #TODO: fix this
				break
			empty = (np.sum([cell.state[2] for cell in self.cells]) < MACHINE_PRECISION)


		# expected latency cannot be less than the free-flow latency of the whole road
		latency = np.clip(latency, np.sum([1.0/cell.vf for cell in self.cells]), np.inf)

		self.cells = temp_cells

		return latency

	
	# value function for the reinforcement learning (= minus cost).
	def value(self):

		# states are now normalized to total number of vehicles, not densities
		total_cars = np.sum([cell.state[2] for cell in self.cells])

		return -total_cars

		
	def reset(self):
		for i in range(self.num_cells):
			self.cells[i].reset()
			

	def randomize_state(self):
		self.reset()

		dens = np.zeros( self.num_cells )
		aut = np.zeros( self.num_cells )

		# randomize densities in each cell and randomize autonomy level in each cell
		# for overall density, draw from uniform distribution between 0 and jam density
		# for autonomy level, draw from uniform distribution between 0 and 1
		# inputs: low, high, size
		for i in range(self.num_cells):
			aut[i] = self.np_random.uniform(0.0, 1.0, 1)
			dens[i] = self.np_random.uniform(0.0,self.cells[i].critdens(aut[i])*1.2, 1)
			self.cells[i].state[0] = dens[i]*self.cells[i].vf*(1 - aut[i])
			self.cells[i].state[1] = dens[i]*self.cells[i].vf*aut[i]
			self.cells[i].state[2] = self.cells[i].state[0] + self.cells[i].state[1]
			self.cells[i].state[3] = aut[i]

	def get_states_for_vis(self):
		# find critical density for each cell
		crit_densities = np.array([cell.critdens(autlevel=cell.state[3]) for cell in self.cells])
		congested = (np.array([cell.state[2] for cell in self.cells]) > crit_densities + MACHINE_PRECISION)*1.0

		# return np.array([cell.state[2] / cell.cell_length for cell in self.cells]), np.array([cell.state[3] for cell in self.cells]), congested

		return np.array([cell.state[2] for cell in self.cells]), np.array([cell.state[3] for cell in self.cells]), congested, crit_densities

	def seed(self, seed):
		self.np_random, seed = seeding.np_random(seed)
		for c in self.cells:
			c.seed(self.np_random.randint(1000))
		return [seed]


# in ParallelNetwork (unlike in Road), all densities are in vehicles per meter. this allows us to normalize observations
class ParallelNetwork(gym.Env):
	def __init__(self, P=3, start_empty=False, demand=[1.993974,2.990961], init_learn_rate=0.5, constant_learn_rate=True, sim_duration=5.0, start_from_equilibrium=False, accident_param=0.6, demand_noise_std=[1.993974/10,2.990961/10]):
		self.sim_duration = sim_duration*secperhr
		self.start_empty = start_empty
		self.start_from_equilibrium = start_from_equilibrium and not start_empty
		self.num_paths = P
		self.init_learn_rate = init_learn_rate
		self.constant_learn_rate = constant_learn_rate
		self.demand = demand
		self.accident_param = accident_param
		self.demand_noise_std = demand_noise_std
		self.initialize()

	def initialize(self):
		self.max_step_size = int(self.sim_duration/T)
		
		if self.num_paths >= 2:
			import gym_trafficnetwork.envs.utils as utils
			road_set = []
			# Do not use [Cell]*3, because we need deep copy
			
			road_set.append(utils.two_partition_road(int(10/quantization),int(5/quantization),60*kmpermiles,1*kmpermiles*1000*quantization,3,2))
			road_set.append(utils.two_partition_road(int(12/quantization),int(4/quantization),75*kmpermiles,1.25*kmpermiles*1000*quantization,4,3))
			for _ in range(self.num_paths-2):
				road_set.append(utils.two_partition_road(int(16/quantization),int(4/quantization),75*kmpermiles,1.25*kmpermiles*1000*quantization,4,3))
			# expected number of accidents in one hour = self.accident_param
			p = self.accident_param*(self.sim_duration/secperhr) / (float(self.max_step_size) * np.sum([cell.num_lanes for road in road_set for cell in road]))
			self.roads = [Road(road, p) for road in road_set]
		else:
			sys.exit('We don\'t have such an experiment: P must be an integer larger than 1. Otherwise, please edit the environment file accordingly.')

		self.max_num_cells = max([road.num_cells for road in self.roads])
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
		
		# use a queue to keep track of how many vehicles are waiting to join the network
		self.q = Vehs(0., 0.)
		
		# reward will be the difference between two value functions (of consecutive timesteps)
		self.last_value = 0
		
	def set(self, property, value):
		if property.lower() == 'sim_duration':
			assert(value >= 0)
			self.sim_duration = value*secperhr
		elif property.lower() == 'start_empty':
			self.start_empty = value
		elif property.lower() == 'start_from_equilibrium':
			self.start_from_equilibrium = value
		elif property.lower() == 'p':
			self.num_paths = value
		elif property.lower() == 'init_learn_rate':
			assert(value >= 0)
			self.init_learn_rate = value
		elif property.lower() == 'constant_learn_rate':
			self.constant_learn_rate = value
		elif property.lower() == 'demand':
			assert(np.min(value) >= 0)
			self.demand = value
		elif property.lower() == 'accident_param':
			assert(value >= 0)
			self.accident_param = value
		elif property.lower() == 'demand_noise_std':
			assert(np.min(value) >= 0)
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
		elif property.lower() == 'p':
			return self.num_paths
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
		
	@property
	def action_space(self):
		if self.num_paths == 2:
			return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
		return spaces.Box(low=0, high=1, shape=(self.num_paths,), dtype=np.float32)
	
	@property
	def observation_space(self):

		# observation should include densities of all cells, as well as the number of human-driven and autonomous
		# vehicles in the queue. TODO: also include the autonomous vehicle demand
		dim_cells = 2*np.sum([road.num_cells for road in self.roads])
		dim_queues = 2
		
		if self.accident_param > MACHINE_PRECISION:
			dim_lanes = np.sum([cell.num_lanes for road in self.roads for cell in road.cells])
			low = np.concatenate((np.zeros((dim_cells,)), np.zeros((dim_queues,)), np.zeros((dim_lanes,))))
			high = np.concatenate((np.concatenate([[[cell.nj/cell.cell_length]*2 for cell in road.cells] for road in self.roads]).reshape(-1), np.ones((dim_queues,))*np.inf, np.ones((dim_lanes,))*np.inf))
		else:
			low = np.concatenate((np.zeros((dim_cells,)), np.zeros((dim_queues,))))
			high = np.concatenate((np.concatenate([[[cell.nj/cell.cell_length]*2 for cell in road.cells] for road in self.roads]).reshape(-1), np.ones((dim_queues,))*np.inf))

		return spaces.Box(low=low, high=high, dtype=np.float32)
		
	def step(self, action, no_accidents=False):

		num_h, num_a = self.get_demand()
		
		# first add the arriving vehicles to the back of the queue (append right, pop left)
		if num_a + num_h > MACHINE_PRECISION:
			new_volume = self.q.volume + num_h + num_a
			self.q = Vehs(new_volume, (self.q.aut_lev*self.q.volume + num_a)/new_volume)
		
		self.human_distribution, self.n_t_h = self.set_selfish_decision(self.human_distribution, self.n_t_h)
	
		action = np.clip(action, self.action_space.low, self.action_space.high)
		if self.num_paths == 2:
			action = np.append(action[0], 1-action[0])
		else:
			if np.all(action < MACHINE_PRECISION):
				action = np.array([1.]*self.num_paths)
			action = action / action.sum()
		# At this point, action is a vector whose sum is 1 and that consists of num_paths entries
		
		# This part used to solve the optimization for 'allocating packets of vehicles in a queue'
		w = self.q.volume
		alpha = self.q.aut_lev
		for j in range(self.num_paths):
			constraint_satisfied = w*alpha*action[j] + w*(1-alpha)*self.human_distribution[j] <= self.roads[j].max_incoming()
			if not constraint_satisfied:
				break
		if not constraint_satisfied:
			bound_on_w = [np.inf]*self.num_paths
			for j in range(self.num_paths):
				denom = alpha*action[j] + (1-alpha)*self.human_distribution[j]
				if denom > MACHINE_PRECISION:
					bound_on_w[j] = self.roads[j].max_incoming() / denom
			w = np.min(bound_on_w)

		# update the queue
		new_volume = self.q.volume - w
		if new_volume > MACHINE_PRECISION:
			self.q = Vehs(new_volume, (self.q.aut_lev*self.q.volume - w*alpha)/new_volume)
		else:
			self.q = Vehs(0., 0.)


		num_h = np.sum(w*(1-alpha))
		num_a = np.sum(w*alpha)
		value = 0
		for i in range(self.num_paths):
			value += self.roads[i].step(num_h*self.human_distribution[i], num_a*action[i], no_accidents)

		# count the vehicles in the queue
		value -= self.q.volume
		
		# take the difference in the value function as the reward
		reward = value - self.last_value
		self.last_value = value
		
		#print(value)
		self.step_count += 1
		return self._get_obs(), reward, self.step_count>=self.max_step_size, {}
		
	def _get_obs(self):
		# get cell ad queue states
		cell_states = np.array([np.array(cell.state[:2]) / cell.cell_length for road in self.roads for cell in road.cells]).reshape(-1)
		queue_state = np.array(self.get_queue_summary())
		if self.accident_param > MACHINE_PRECISION:
			lane_states = np.array([a for road in self.roads for cell in road.cells for a in cell.accidents])
			lane_state = (lane_states > 0).astype(float) # the agent does not see how long it will take to clear out the accidents
			return np.concatenate((cell_states, queue_state, lane_states))
		else:
			return np.concatenate((cell_states, queue_state))
	
	# returns the number of human-driven and autonomous vehicles in the queue
	def get_queue_summary(self):
		return self.q.volume*(1.0-self.q.aut_lev), self.q.volume*self.q.aut_lev
		
	def reset(self):
		# reset preferences
		self.human_distribution = np.array([1.0/self.num_paths]*self.num_paths)
		self.n_t_h = self.init_learn_rate 
		for road in self.roads:
			road.reset()

		if not self.start_empty:
			# then randomly set densities
			for road in self.roads:
				road.randomize_state()
			if self.start_from_equilibrium:
				self.go_to_equilibrium(self.T_init)
		self.step_count = 0
		self.q = Vehs(0., 0.)
		self.last_value = 0

		return self._get_obs()
		

	def go_to_equilibrium(self, T):
		aut_distribution = np.array([1.0/self.num_paths]*self.num_paths)
		n_t_a = self.init_learn_rate
		for _ in range(T):
			aut_distribution, n_t_a = self.set_selfish_decision(aut_distribution, n_t_a)
			self.step(aut_distribution, no_accidents=True)

	def close(self):
		pass
		
	def seed(self, seed):
		self.np_random, seed = seeding.np_random(seed)
		for rd in self.roads:
			rd.seed(self.np_random.randint(1000))
		return [seed]
		
	# generic update function for either class of vehicle. pass in current distributions
	# and learning rate and it returns the updated distribution and learning rate
	def set_selfish_decision(self, dist, n_t):
		assert( dist.size == self.num_paths )
		assert( n_t >= 0 )

		new_dist = dist*0
		latencies = [road.measure_latency() for road in self.roads]
		for p in range(self.num_paths):
			numerator = dist[p]*np.exp(-n_t*latencies[p])
			denominator = np.sum([dist[q]*np.exp(-n_t*latencies[q]) for q in range(self.num_paths)])
			new_dist[p] = numerator / denominator

		# update learning rate
		new_n_t = n_t
		if not self.constant_learn_rate:
			new_n_t = 1.0/(1.0/n_t + 1) # update learning rate from 1/t to 1/(t+1)
		
		return new_dist, new_n_t
		
	def get_demand(self):
		return max(self.demand_h + self.demand_noise_std[0]*self.np_random.randn(),0), max(self.demand_a + self.demand_noise_std[1]*self.np_random.randn(),0)
		#return self.demand_h, self.demand_a

# keeps track of vehicles. inputs: volume, autonomy level
# stored values: volume, autonomy level.
# TODO: use named tuple instead?
class Vehs:
	def __init__(self, volume, aut_lev):
		self.volume = volume
		self.aut_lev = aut_lev
