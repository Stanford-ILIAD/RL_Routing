from gym_trafficnetwork.envs.parallel_network import Cell
import numpy as np

# For the simplest road type
def homogeneous_road(num_cells, vfkph, cell_length, num_lanes):
	r = []
	for _ in range(num_cells):
		r.append(Cell(vfkph, cell_length, num_lanes))
	return r
	
# For roads who have cells with the number of lanes as n-n-n-m-n
def road_with_single_bottleneck(num_cells, vfkph, cell_length, num_lanes, bottleneck_id, bottleneck_num_lanes):
	# bottleneck_id is the id of the cell that has bottleneck_num_lanes-many lanes (0 is the first cell, and num_cells-1 is the last)
	
	# I know we will say "let's we have 5 cells and the last one is the bottleneck, so bottleneck_id is 5". Let's correct it.
	if bottleneck_id >= num_cells:
		import warnings
		warnings.warn("bottleneck_id is invalid! I am setting it to be the last cell.")
		import time
		time.sleep(5)
		bottleneck_id = num_cells - 1
	
	r = []
	for _ in range(num_cells - 1):
		r.append(Cell(vfkph, cell_length, num_lanes))
	r.insert(bottleneck_id, Cell(vfkph, cell_length, bottleneck_num_lanes))
	return r

# For roads who have cells with the number of lanes as n-n-n-m-m	
def two_partition_road(firstpart_num_cells, secondpart_num_cells, vfkph, cell_length, firstpart_num_lanes, secondpart_num_lanes):
	r = []
	for _ in range(firstpart_num_cells):
		r.append(Cell(vfkph, cell_length, firstpart_num_lanes))
	for _ in range(secondpart_num_cells):
		r.append(Cell(vfkph, cell_length, secondpart_num_lanes))
	return r
	
# Generalization of the two_partition_road (and homogeneous_road) to n-partition roads. All parameters will be either an array or a scalar
def n_partition_road(num_cells, vfkph, cell_length, num_lanes):
	if not (isinstance(num_cells, list) or isinstance(num_cells, np.ndarray)):
		num_cells = [num_cells]
	if not (isinstance(vfkph, list) or isinstance(vfkph, np.ndarray)):
		vfkph = [vfkph]
	if not (isinstance(cell_length, list) or isinstance(cell_length, np.ndarray)):
		cell_length = [cell_length]
	if not (isinstance(num_lanes, list) or isinstance(num_lanes, np.ndarray)):
		num_lanes = [num_lanes]
	
	num_partitions = np.max([len(num_cells), len(vfkph), len(cell_length), len(num_lanes)])
	if len(num_cells) == 1:
		num_cells = [num_cells[0]]*num_partitions
	if len(vfkph) == 1:
		vfkph = [vfkph[0]]*num_partitions
	if len(cell_length) == 1:
		cell_length = [cell_length[0]]*num_partitions
	if len(num_lanes) == 1:
		num_lanes = [num_lanes[0]]*num_partitions
	
	r = []
	for i in range(len(num_cells)):
		for _ in range(num_cells[i]):
			r.append(Cell(vfkph[i], cell_length[i], num_lanes[i]))
	return r