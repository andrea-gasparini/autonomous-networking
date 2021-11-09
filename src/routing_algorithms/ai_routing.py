
import numpy as np
from typing import Dict, List, Set
from src.entities.uav_entities import DataPacket, Drone
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

class AIRouting(BASE_routing):

	POSITIVE_REWARD: int = 10
	NEGATIVE_REWARD: int = -1

	EPSILON: float = 0.70 # for epsilon-greedy (higher favors exploitation)

	DEBUG: bool = False

	def __init__(self, drone: Drone, simulator):
		BASE_routing.__init__(self, drone, simulator)

		# random generator
		self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)

		self.q_table: Dict[int, Dict[Drone, int]] = dict()				# {cell_index: {drone: action_value}}
		self.n_table: Dict[int, Dict[Drone, int]] = dict()				# {cell_index: {drone: taken_count}}
		self.taken_actions: Dict[int, Dict[int, Set[Drone]]] = dict()	# {id_event: {cell_index: taken_actions}}
		self.taken_actions_energy: Dict[int, Dict[int, float]] = dict()	# {id_event: {cell_index: energy}}


	def feedback(self, drone: Drone, id_event: int, delay: int, outcome) -> None:
		""" return a possible feedback, if the destination drone has received the packet """
		# Packets that we delivered and still need a feedback
		if self.DEBUG: print(self.drone.identifier, "----------", self.taken_actions)
		
		# Feedback from a delivered or expired packet
		if self.DEBUG: print(self.drone.identifier, "----------", drone, id_event, delay, outcome)

		if drone == self.drone: return None # skip the feedback if we are the same drone.

		cell_index = self.__get_cell_index_for_feedback(drone, id_event)

		if cell_index == -1: return None

		# outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
		reward = self.NEGATIVE_REWARD if outcome == -1 else self.POSITIVE_REWARD
		
		if drone not in self.q_table[cell_index]: self.q_table[cell_index][drone] = 0

		self.q_table[cell_index][drone] += (1 / self.n_table[cell_index][drone]) * (reward - self.q_table[cell_index][drone])

		# Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
		# NOTE: reward or update using the old action!!
		# STORE WHICH ACTION DID YOU TAKE IN THE PAST.
		# do something or train the model (?)
		if id_event in self.taken_actions:
			action = self.taken_actions[id_event]
			del self.taken_actions[id_event]


	def relay_selection(self, opt_neighbors: List[Drone], pkd: DataPacket) -> Drone:
		""" arg min score  -> geographical approach, take the drone closest to the depot """
		
		packet_id_event: int = pkd.event_ref.identifier
		# Only if you need --> several features:
		cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
														width_area=self.simulator.env_width,
														x_pos=self.drone.coords[0],  # e.g. 1500
														y_pos=self.drone.coords[1])[0]  # e.g. 500

		# we save the residual energy for the feedback at the end
		if packet_id_event not in self.taken_actions_energy:
			self.taken_actions_energy[packet_id_event] = {cell_index: self.drone.residual_energy}
		else:
			self.taken_actions_energy[packet_id_event][cell_index] = self.drone.residual_energy
		
		if cell_index not in self.q_table:
			self.q_table[cell_index] = dict() # cell_index represents the state, in our case is the drone's cell (index) location.
			self.n_table[cell_index] = dict()

		if not self.q_table[cell_index] or self.rnd_for_routing_ai.random() > self.EPSILON:
			# we do exploration if the q_table[cell_index] is empty or with the e-greedy strategy
			action = self.__exploration(neighbors=opt_neighbors, id_event=packet_id_event, cell_index=cell_index)
		else:
			action = self.__exploitation(neighbors=opt_neighbors, id_event=packet_id_event, cell_index=cell_index)
		
		if action != None:
			if action not in self.n_table[cell_index]:
				self.n_table[cell_index][action] = 1
			else:
				self.n_table[cell_index][action] += 1

		# self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
		# self.drone.residual_energy (that tells us when I'll come back to the depot).
		#  .....

		# Store your current action --- you can add several stuff if needed to take a reward later
		# self.taken_actions[pkd.event_ref.identifier] = (action)
		return action


	def print(self):
		"""
			This method is called at the end of the simulation, can be usefull to print some
				metrics about the learning process
		"""
		pass


	def __get_cell_index_for_feedback(self, drone: Drone, id_event: int) -> int:
		if id_event not in self.taken_actions:
			return -1

		cells_in_id_event = list(self.taken_actions[id_event].keys())
		
		if len(cells_in_id_event) == 1:
			return cells_in_id_event[0]
		
		cells_visited_by_the_drone = []
		for cell in cells_in_id_event:
			if drone in self.taken_actions[id_event][cell]:
				cells_visited_by_the_drone.append(cell)
		
		best_cell = None
		best_energy = -1
		for cell in cells_visited_by_the_drone:
		
			if self.taken_actions_energy[id_event][cell] > best_energy:
				best_cell = cell
				best_energy = self.taken_actions_energy[id_event][cell]
			
		return best_cell


	def __save_action(self, action: Drone, id_event: int = 0, cell_index: int = 0) -> None:
		if id_event not in self.taken_actions:
			self.taken_actions[id_event] = {cell_index: {action}} # create the key for this id_event with this cell and this drone
		else:
			if cell_index not in self.taken_actions[id_event]:
				self.taken_actions[id_event][cell_index] = {action} # the id event was in the taken actions but not on this cell, so add the cell
			else:
				self.taken_actions[id_event][cell_index].add(action) # otherwise, we just add the new Drone selected


	def __exploration(self, neighbors: List[Drone] = [], id_event: int = 0, cell_index: int = 0) -> Drone:
		""" Choose an action for exploration """
		neighbors_drones: Set[Drone] = {drone[1] for drone in neighbors} # Set of neighbors Drone objects
		action: Drone = None

		if id_event in self.taken_actions:
			if cell_index in self.taken_actions[id_event]:
				already_taken_drones: Set[Drone] = self.taken_actions[id_event][cell_index]
				if len(list(neighbors_drones - already_taken_drones)) == 0:
					return action
				else:
					action = self.rnd_for_routing_ai.choice(list(neighbors_drones - already_taken_drones)) # we take a drone randomly, this is exploration
		else:
			action = self.rnd_for_routing_ai.choice(list(neighbors_drones)) # we take a drone randomly, this is exploration

		# TODO FIXME here action can still be None
		self.__save_action(action=action, id_event=id_event, cell_index=cell_index)
		return action

		
	def __exploitation(self, neighbors: List[Drone] = [], id_event: int = 0, cell_index: int = 0) -> Drone:

		best_drone, best_drone_action_value = None, None
		
		for drone, action_value in self.q_table[cell_index].items():
			if best_drone is None or action_value > best_drone_action_value:
				best_drone, best_drone_action_value = drone, action_value

		if best_drone in neighbors:
			self.__save_action(action=best_drone, id_event=id_event, cell_index=cell_index)
			return best_drone
