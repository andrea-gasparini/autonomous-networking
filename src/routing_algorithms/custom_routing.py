
import numpy as np
from typing import Dict, List, Set, Tuple
from src.entities.uav_entities import DataPacket, Drone
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

class CUSTOMRouting(BASE_routing):

	ALPHA = 1.5

	def __init__(self, drone: Drone, simulator):
		BASE_routing.__init__(self, drone, simulator)
		# random generator
		self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
		self.taken_actions: Dict[int, Tuple[int, Drone]] = dict()  # {id event: (state, old_action)}

		'''
			A bit of theory:
				- Q*(s, a) = expected value of doing a in state s and then following the optimal policy
				- Temporal differences to estimate the value of Q*(s, a) -> It is an agent learning from an environment through episodes with no prior knowledge of the environment.
				- The agent maintains a table of Q[S, A] where S is the set of states and A is the set of actions.
				- Q[s, a] represents its current estimate of Q*(s, a)
			
			In this case, we have that the state is represented by the drone's positions (number of cells that the drone will traverse) and the possible action is to transfer the packet or not (so 2).
		'''
		self.q_table = [[0 for i in range(len(self.drone.path))] for j in range(2)]
		self.count_action_table: Dict[Drone, int] = {} # {action: count}

		self.force_exploration = True # to force the exploration
		self.epsilon = 0.02 # the epsilon-greedy implementation


	def feedback(self, drone: Drone, id_event: int, delay: int, outcome) -> None:
		
		if id_event in self.taken_actions:
			state, action = self.taken_actions[id_event]

			# delete the id_evnet
			del self.taken_actions[id_event]
			
			# it is used in the self.q_table to check the action (0 if the drone DIDN'T pass the packet, 1 otherwise)
			pass_packet: bool = action == self.drone # check if the action is to pass or not (if action == self.drone then I do not pass the packet)
			
			if outcome == -1:
				reward = -1
			else:
				reward = 0.05 * delay * (action.residual_energy / self.simulator.drone_max_energy) # random reward based on delay and residual energy
			
			# just update the Q-table, play with these values to see if something
			# TODO: we can change also the update function as in wiki (https://it.wikipedia.org/wiki/Q-learning) adding a gamma and the alpha
			self.q_table[pass_packet][state] += 1 / self.count_action_table[action] * (reward - self.q_table[pass_packet][state])
			

	def _update_count_action_table(self, drone: Drone):
		if drone not in self.count_action_table:
			self.count_action_table[drone] = 1
		else:
			self.count_action_table[drone] += 1


	def _select_best_drone(self, neighbors: List[Drone], state: int) -> Drone:
		best_drone = None
		best_score = -1
		me_depot_distance = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)
		for drone in neighbors:
			# I take a drone by the best drone score (0.03 is to tuning); ho messo 0.2 ora, prima era 0.03 e arrivavo a 0.6 di delivery ratio
			# I do 0.03 * residual energy of the drone I'm considering multiplied by the euclidean distance between the drone and the simulator depot.
			drone = drone[1]
			pass_packet: bool = drone == self.drone
			drone_depot_distance = util.euclidean_distance(drone.coords, self.simulator.depot_coordinates)
			drone_score = (self.ALPHA * (drone.residual_energy / self.simulator.drone_max_energy) * drone_depot_distance)

			if drone_score > 0:
				drone_score = np.power(drone_score, (pass_packet * self.q_table[0][state] + (1 - pass_packet) * self.q_table[1][state]))

			# if I am closer to the depot, skip this drone
			if me_depot_distance < drone_depot_distance:
				continue

			if drone_score > best_score:
				best_score = drone_score
				best_drone = drone

		return best_drone if best_drone != None else self.drone


	def relay_selection(self, opt_neighbors: List[Drone], pkd: DataPacket):

		neighbors_drones_instances: Set[Drone] = {drone[1] for drone in opt_neighbors} # all neighbors drones
		neighbors_drones_instances.add(self.drone) # add me in order to the possible choice (so, if we select self.drone we are keeping the packet)
		state = self.drone.path.index(self.drone.next_target()) # get the state

		# we need to calculate the action!
		if self.force_exploration or self.rnd_for_routing_ai.rand() < self.epsilon:
			# we are doing exploration, so select RANDOMLY from the neighbors_drones_instances.
			action = self.rnd_for_routing_ai.choice(list(neighbors_drones_instances))
			self.force_exploration = False # we reset the force_exploration for the first run.
			self._update_count_action_table(action)
		else:
			# here we have to take the best drone, for our policy!
			action = self._select_best_drone(opt_neighbors, state)
			self._update_count_action_table(action)

		self.taken_actions[pkd.event_ref.identifier] = (state, action) # save the taken action for the state

		return action


	def print(self):
		"""
			This method is called at the end of the simulation, can be usefull to print some
				metrics about the learning process
		"""
		pass
