import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

MACHINE_PRECISION = 1e-10

secperhr = 60*60
T = 1/100*secperhr # seconds per time step (1/100 of an hour to travel 1km at 100kph)
tauh = 2.0 # human reaction time
taua = 1.0 # aut reaction time
L = 4.0 # car length (meters)
nj = 1.0/L # jam density, assume it's the same on all roads

class Road:
	def __init__(self, vfkph, num_cells, cell_dist_meters):
		self.vfms = vfkph*1000.0/secperhr # free-flow speed in meters per second
		self.headway = [self.vfms * tauh, self.vfms * taua] # meters
		self.num_cells = num_cells
		self.cell_dist = cell_dist_meters # meters
		self.vf = cell_dist_meters # free-flow speed in meters per timestep
		
		self.reset()
		
	# arguments: human and autonomous incoming flows (number of vehicles),
	# Note that road length should be equal to vf in our formulation
	def step(self, num_h, num_a):
		self.arrival_rate = num_h + num_a
	
		fout = np.zeros_like(self.state[:,0]) # create an array that will hold the output flow from each cell
		
		maxIncoming = np.inf # for final cell there is no limit on how much can be output.
		# start at the last cell and work our way backwards
		for i in range(len(self.state)-1,-1,-1):
			# first find flow exiting from the cell being considered
			nc = self.critdens(self.state[i,3])
			w = self.shockspeed(nc)
			
			# find the volume of flow exiting the cell
			fout[i] = np.min([self.state[i,2]*self.vf, w*(nj-self.state[i,2]), maxIncoming])
			
			# now update the last cell accordingly (autonomy level remains unchanged)
			self.state[i,0] -= (1-self.state[i,3]) * fout[i] / self.cell_dist
			self.state[i,1] -= self.state[i,3] * fout[i] / self.cell_dist
			self.state[i,2] = self.state[i,0] + self.state[i,1]

			# if it is not the final cell, update the following cell with the incoming flow
			if i < len(self.state)-1:
				# autonomy level may change
				self.state[i+1,0] += (1-self.state[i,3]) * fout[i] / self.cell_dist
				self.state[i+1,1] += self.state[i,3] * fout[i] / self.cell_dist
				self.state[i+1,2] = self.state[i+1,0] + self.state[i+1,1]
				if self.state[i+1,2] > MACHINE_PRECISION:
					self.state[i+1,3] = self.state[i+1,1] / self.state[i+1,2]
					self.state[i+1,3] = np.clip(self.state[i+1,3], 0, 1) # numerical issues causes -1e-16 like things...
				else:
					self.state[i+1,3] = 0
			maxIncoming = (nj - self.state[i,2])*self.cell_dist

		self.exploded = (num_h + num_a > maxIncoming)

		# now add input to first cell, first making sure that the cell has capacity
		self.state[0,0] += num_h / self.cell_dist
		self.state[0,1] += num_a / self.cell_dist
		self.state[0,2] = self.state[0,0] + self.state[0,1]
		if self.state[0,2] > MACHINE_PRECISION:
			self.state[0,3] = self.state[0,1] / self.state[0,2]
			self.state[0,3] = np.clip(self.state[0,3], 0, 1) # numerical issues causes -1e-16 like things...
		else:
			self.state[i+1,3] = 0
		
		return self.reward()

	# this function serves as an estimate of the flow rate exiting each cell, which humans use 
	# to estimate road latencies
	def estimate_fout(self):
		fout_est = np.zeros_like(self.state[:,0]) # create an array that will hold the output flow from each cell
		maxIncoming = np.inf # for final cell there is no limit on how much can be output.
		# start at the last cell and work our way backwards
		for i in range(len(self.state)-1,-1,-1):
			# first find flow exiting from the cell being considered
			nc = self.critdens(self.state[i,3])
			w = self.shockspeed(nc)
			# find the volume of flow exiting the cell.
			# since we use this to estimate how many time steps it would take incoming flow to exit,
			# our output in the uncongested regime (first argument) is limited only by the maximum flow
			fout_est[i] = np.min([nc*self.vf, w*(nj-self.state[i,2]), maxIncoming])	
			maxIncoming = (nj - self.state[i,2])*self.cell_dist
		return fout_est
		
	# human driver's estimation of the latency on a road. corresponds to how long it 
	# would take to traverse a road if densities and cell flows remained constant.
	# arguments: road length, vector of flows exiting each cell
	def estimate_latency(self):
		fout_est = self.estimate_fout()
		latency = 0
		for i in range(len(self.state)):
			if i==0:
				q = self.state[i,2]
				latency += np.ceil(q/fout_est[i])
			else:
				q = self.state[i,2] * self.cell_dist - (fout_est[i-1] - (q % fout_est[i-1]))
				latency += np.ceil(q/fout_est[i])
		return np.clip(latency, 1, np.inf)
	
	# reward function for the reinforcement learning (= minus cost).
	def reward(self):
		if not self.exploded:
			if self.arrival_rate == 0:
				return 0
			total_cars = self.state[:,2].sum()*self.cell_dist
			return 15-total_cars/self.arrival_rate
		return -1

	# inputs: autonomy level, array of headways [human, autonomous], vehicle length, road length
	def critdens(self, autlevel):
		assert(autlevel >= 0)
		assert(autlevel <= 1)
		return 1.0 /(autlevel*self.headway[1] + (1-autlevel)*self.headway[0] + L)

	# velocity in meters per timestep
	def maxflow(self, critdensity):
		return critdensity*self.vf
	
	def shockspeed(self, critdensity):
		return critdensity*self.vf/(nj - critdensity)
		
	def reset(self):
		self.state = np.zeros(shape=(self.num_cells,4))
		self.arrival_rate = 0
		self.exploded = False # did we explode yet?
		
class RoadNetwork(gym.Env):
	def __init__(self, num_roads=2, vfkph=[50,100], num_cells=[3,5], cell_dist_meters=[500,1000]):
		assert(num_roads >= 2)
		assert(len(vfkph) == num_roads)
		assert(len(num_cells) == num_roads)
		assert(len(cell_dist_meters) == num_roads)
		self.roads = [Road(vfkph[i], num_cells[i], cell_dist_meters[i]) for i in range(num_roads)]
		self.reward_range = (-np.inf, np.inf)
		self.step_count = 0
		
		self.human_distribution = np.array([1.0/num_roads]*num_roads)
		self.n_t = 1 # initial learning rate for humans
		
	@property
	def action_space(self):
		dim = len(self.roads)
		if dim == 2:
			return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
		return spaces.Box(low=0, high=1, shape=(dim,), dtype=np.float32)
	
	@property
	def observation_space(self):
		dim = 2*np.sum([self.roads[i].num_cells for i in range(len(self.roads))])
		return spaces.Box(low=0, high=nj, shape=(dim,), dtype=np.float32)
		
	def step(self, action):
		num_h, num_a = self.get_demand()
		self.set_human_action() # will give a vector whose sum is 1
	
		num_roads = len(self.roads)
		action = np.clip(action, self.action_space.low, self.action_space.high)
		if num_roads == 2:
			action = np.append(action[0], 1-action[0])
		else:
			action = action / action.sum()
		
		# At this point, action is a vector whose sum is 1 and that consists of num_roads entries
		reward = 0
		for i in range(len(self.roads)):
			reward += self.roads[i].step(num_h*self.human_distribution[i], num_a*action[i])
		
		#print(reward)
		self.step_count += 1
		return self._get_obs(), reward, reward<0 or self.step_count>=300, {}
		
	def _get_obs(self):
		return np.concatenate([road.state[:,:2] for road in self.roads]).reshape(-1)
		
	def reset(self):
		self.step_count = 0
		for road in self.roads:
			road.reset()
		return self._get_obs()
	
	def render(self):
		print(self._get_obs())
			
	def close(self):
		pass
		
	def seed(self, seed):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def set_human_action(self):
		self.n_t = 1.0/(1.0/self.n_t + 1) # update it from 1/t to 1/(t+1)
		new_dist = self.human_distribution*0
		latencies = [road.estimate_latency() for road in self.roads]
		for p in range(len(self.roads)):
			numerator = self.human_distribution[p]*np.exp(-self.n_t*latencies[p])
			denominator = np.sum([self.human_distribution[q]*np.exp(-self.n_t*latencies[q]) for q in range(len(self.roads))])
			new_dist[p] = numerator / denominator
		self.human_distribution = new_dist
		
	def get_demand(self):
		return 15 + 2*self.np_random.randn(), 15 + 2*self.np_random.randn()