
import numpy as np
from typing import Dict, List, Set
from src.entities.uav_entities import DataPacket, Drone
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

class OptimisticAIRouting(BASE_routing):

	def __init__(self, drone: Drone, simulator):
		BASE_routing.__init__(self, drone, simulator)

		# random generator
		self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
		self.taken_actions = {}

		# i = 0 -> keep packet, i = 1 pass packet
		self.q_table: List[int] = [5 for i in range(self.simulator.n_drones)] # Q-table for the two actions: keep or pass the packet.
		self.n_table: List[int] = [0 for i in range(self.simulator.n_drones)] # N-table for the count of the two actions.
		self.epsilon: int = 0.02 # [0.030, 0.040], 0.2 abbiamo 0.72
		self.force_exploration = True
		self.alpha = 1.5
		self.exploration, self.exploitation = 0, 0
		self.drone_explored: Set[Drone] = set()
		self.feedback_timestep = 0
		self.rewards = 0
		self.avg_rewards = []


	def feedback(self, drone: Drone, id_event: int, delay: int, outcome) -> None:
		""" return a possible feedback, if the destination drone has received the packet """
		# Packets that we delivered and still need a feedback
		# Drone is the drone that wants to transfer packets OR it is the drone that has lost some packets
		
		# https://imgur.com/a/JqjB0gP

		# Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
		# NOTE: reward or update using the old action!!
		# STORE WHICH ACTION DID YOU TAKE IN THE PAST.
		# do something or train the model (?)
		if id_event in self.taken_actions:
			action: Drone = self.taken_actions[id_event]
			if outcome == -1:
				reward = -2
			else:
				reward = 2 + (2 * (action.residual_energy / action.max_energy) / delay) # random reward 
			
			del self.taken_actions[id_event]
			self.q_table[action.identifier] += 1 / self.n_table[action.identifier] * (reward - self.q_table[action.identifier])

			self.rewards += reward
			self.feedback_timestep += 1
			self.avg_rewards += [self.rewards / self.feedback_timestep]


	def relay_selection(self, opt_neighbors: List[Drone], pkd: DataPacket) -> Drone:
		""" arg min score  -> geographical approach, take the drone closest to the depot """
		
		packet_id_event: int = pkd.event_ref.identifier
		# Only if you need --> several features:
		#cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
		#												width_area=self.simulator.env_width,
		#												x_pos=self.drone.coords[0],  # e.g. 1500
		#					Removed							y_pos=self.drone.coords[1])[0]  # e.g. 500

		action = None

		neighbors_drones: Set[Drone] = {drone[1] for drone in opt_neighbors}

		if packet_id_event not in self.taken_actions:
			self.taken_actions[packet_id_event] = None

		if self.force_exploration or self.rnd_for_routing_ai.rand() < self.epsilon:
			neighbors_drones.add(self.drone)
			action = self.rnd_for_routing_ai.choice(list(neighbors_drones))
			self.force_exploration = False
			self.taken_actions[packet_id_event] = action
			self.exploration += 1
		else:
			action = self.__exploitation(neighbors_drones)			
			self.taken_actions[packet_id_event] = action
			self.exploitation += 1
			
		self.n_table[action.identifier] += 1

		# self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
		# self.drone.residual_energy (that tells us when I'll come back to the depot).
		#  .....

		# Store your current action --- you can add several stuff if needed to take a reward later
		# self.taken_actions[pkd.event_ref.identifier] = (action)
		return action


	def __exploitation(self, neighbors: List[Drone]) -> Drone:
		best_drone = None
		best_score = float('-inf')
		#neighbors_and_distance = [(drone, util.euclidean_distance(drone.next_target(), self.simulator.depot_coordinates), self.q_table[drone.identifier]) for drone in neighbors]
		#neighbors_and_distance_sorted = sorted(neighbors_and_distance, key = lambda x: x[2])[0][0]
		#print(neighbors_and_distance, neighbors_and_distance_sorted, self.drone)
		#return neighbors_and_distance_sorted
		me_depot_distance = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)

		for drone in neighbors:
			# I take a drone by the best drone score (0.03 is to tuning); ho messo 0.2 ora, prima era 0.03 e arrivavo a 0.6 di delivery ratio
			# I do 0.03 * residual energy of the drone I'm considering multiplied by the euclidean distance between the drone and the simulator depot.
			drone_depot_distance = util.euclidean_distance(drone.coords, self.simulator.depot_coordinates)
			
			# if I am closer to the depot, skip this drone
			if me_depot_distance < drone_depot_distance:
				continue

			if drone_depot_distance <= self.simulator.depot_com_range:
				best_drone = drone
				break
			else:
				drone_score = (self.alpha * (drone.residual_energy / drone.max_energy) / drone_depot_distance) * (self.q_table[drone.identifier])

			if drone_score > best_score:
				best_score = drone_score
				best_drone = drone

		return best_drone if best_drone != None else self.drone


	def print(self):
		"""
			This method is called at the end of the simulation, can be usefull to print some
				metrics about the learning process
		"""
		print('-' * 50)
		print('Exploration count', self.exploration)
		print('Exploitation count', self.exploitation)
		steps = np.arange(self.feedback_timestep)
		#plt.plot(steps, self.avg_rewards)
		#plt.ylabel("avg rewards")
		#plt.xlabel("feedback")
		#plt.show()
		print('-' * 50)
