import numpy as np
from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config

class QLearningMGEO(BASE_routing):

	ALPHA: float = 0.5
	GAMMA: float = 0.6
	EPSILON: float = 0.02

	def __init__(self, drone: Drone, simulator) -> None:
		BASE_routing.__init__(self, drone, simulator)
		# random generator
		self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
		self.taken_actions: Dict[int, Tuple[Union[None, Literal[-1], Drone], int, int]] = {}  # {id_event : (old_action, cell_index, next_cell_index)}
		self.drone: Drone = drone

		self.q_table: Dict[int, List[int]] = {} # {0: [0, ..., 0]}
		#self.n_table: Dict[] = {}
		self.force_exploration: bool = True

		self.tot_energy_to_depot = 0


	def feedback(self, drone: Drone, id_event: int, delay: int, outcome: int) -> None:
		""" return a possible feedback, if the destination drone has received the packet """
		if config.DEBUG:
			# Packets that we delivered and still need a feedback
			print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

			# outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
			# Feedback from a delivered or expired packet
			print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
				"Drone:", drone, " - id-event:", id_event, " - delay:",  delay, " - outcome:", outcome)

		# Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
		# NOTE: reward or update using the old action!!
		# STORE WHICH ACTION DID YOU TAKE IN THE PAST.
		# do something or train the model (?)
		if id_event in self.taken_actions:
			action, cell_index, next_cell_index = self.taken_actions[id_event]

			if action == None:
				action = self.drone.identifier
				selected_drone = self.drone
			elif isinstance(action, Drone):
				selected_drone = action
				action = action.identifier
			
			if outcome == -1:
				reward = -2
			else:
				reward = 2

		
			self.q_table[cell_index][action] += self.ALPHA * (reward + self.GAMMA * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action])

			del self.taken_actions[id_event]


	def __exploration(self, neighbors_drones: Set[Drone]) -> Union[None, Literal[-1], Drone]:
		""" Do exploration """
		action = None
		if len(neighbors_drones) > 0:
			action = self.rnd_for_routing_ai.choice(list(neighbors_drones))
		#else:
		#	action = -1
		self.force_exploration = False
		return action

	def __exploitation(self, neighbors_drones: Set[Drone], cell_index: int) -> Union[None, Literal[-1], Drone]:
		best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
		best_drone = None
		# Action MOVE is identified by -1
		if len(neighbors_drones) == 0 and self.drone.buffer_length() >= 1 or self.drone.move_routing:
			return -1

		factor = -1 if not self.drone.move_routing else 1
		tmp_score = factor * best_drone_distance_from_depot / self.drone.speed * self.drone.buffer_length() * self.q_table[cell_index][self.drone.identifier]
		for drone in neighbors_drones:
			factor = -1 if not drone.move_routing else 1
			drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, drone.coords)
			drone_score = factor * drone_distance_from_depot / drone.speed * drone.buffer_length() * self.q_table[cell_index][drone.identifier]
			if drone_score >= tmp_score:
				best_drone = drone
				tmp_score = drone_score
		
		return best_drone
	
	def relay_selection(self, opt_neighbors: List[Tuple[DataPacket, Drone]], pkd: DataPacket) -> Union[None, Literal[-1], Drone]:
		"""
			Three actions are possible:
			- MOVE to depot (return -1)
			- KEEP packet 	(return None)
			- SEND packet 	(return the drone)

			Notice that, if you selected -1, you will move until your buffer is empty.
			But you still need to select a relay drone, if any, to empty your buffer.
		"""
		# Notice all the drones have different speed, and radio performance!!
		# you know the speed, not the radio performance.
		# self.drone.speed

		# Only if you need --> several features:
		cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
														width_area=self.simulator.env_width,
														x_pos=self.drone.coords[0],  # e.g. 1500
														y_pos=self.drone.coords[1])[0]  # e.g. 500
		action = None
		if cell_index not in self.q_table:
			self.q_table[cell_index] = [0 for i in range(self.simulator.n_drones + 1)]

		neighbors_drones: Set[Drone] = {drone[1] for drone in opt_neighbors}
		if self.force_exploration or self.rnd_for_routing_ai.rand() < self.EPSILON:
			neighbors_drones.add(self.drone)
			action = self.__exploration(neighbors_drones)
		else:
			action = self.__exploitation(neighbors_drones, cell_index)
		

		# self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
		# self.drone.residual_energy (that tells us when I'll come back to the depot).
		#  .....

		# Store your current action --- you can add several stuff if needed to take a reward later
		next_cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
														width_area=self.simulator.env_width,
														x_pos=self.drone.next_target()[0],
														y_pos=self.drone.next_target()[1])[0]

		self.taken_actions[pkd.event_ref.identifier] = (action, cell_index, next_cell_index)

		if next_cell_index not in self.q_table:
			self.q_table[next_cell_index] = [0 for i in range(self.simulator.n_drones + 1)]

		return action


	def print(self):
		"""
			This method is called at the end of the simulation, can be useful to print some
			metrics about the learning process
		"""
		pass
