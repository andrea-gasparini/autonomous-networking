import numpy as np
from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config

class AIRouting(BASE_routing):

	ALPHA: float = 0
	"""
	Learning rate
	"""
	GAMMA: float = 0
	"""
	Importance of future rewards
	"""
	EPSILON: float = 0.2
	"""
	Probability of performing exploration
	"""
	EXPLORATION_SEND_PROBABILTY: float = 0.6
	"""
	Probabilty of sending the packet to a random drone during exploration
	"""

	def __init__(self, drone: Drone, simulator) -> None:
		BASE_routing.__init__(self, drone, simulator)
		# random generator
		self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
		self.taken_actions: Dict[int, Tuple[Union[None, Literal[-1], Drone], int, int]] = {}  # {id_event : (old_action, cell_index, next_cell_index)}
		self.drone: Drone = drone

		self.q_table: Dict[int, List[int]] = {} # {0: [0, ..., 0]}
		#self.n_table: Dict[] = {}
		self.force_exploration: bool = True



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
				reward = 2 * self.simulator.event_duration / delay

		
			self.q_table[cell_index][action] += self.ALPHA * (reward + self.GAMMA * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action])

			del self.taken_actions[id_event]


	def __exploration(self, neighbors_drones: Set[Drone]) -> Union[None, Literal[-1], Drone]:
		""" Do exploration """
		action = None

		send_to_rnd_drone: bool = self.rnd_for_routing_ai.rand() > 1 - self.EXPLORATION_SEND_PROBABILTY
		neighbors_returning_to_depot: List[Drone] = [drone for drone in neighbors_drones if drone.move_routing]

		if send_to_rnd_drone and self.drone.buffer_length() == 1:
			action = self.rnd_for_routing_ai.choice(list(neighbors_drones))
		elif self.drone.buffer_length() >= 2:
			action = -1 if len(neighbors_returning_to_depot) == 0 else self.rnd_for_routing_ai.choice(list(neighbors_returning_to_depot))
		else:
			action = None

		self.force_exploration = False

		return action if action != self.drone else None

	
	def __exploitation_1795030(self, neighbors_drones: Set[Drone], cell_index: int) -> Union[None, Literal[-1], Drone]:
		action = None
		drone_score = float('-inf')

		self_depot_distance = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)
		time_self_to_depot = self_depot_distance / self.drone.speed

		if len(neighbors_drones) == 0 and self.drone.buffer_length() > 2:
			return -1

		for drone in neighbors_drones:

			drone_depot_distance = util.euclidean_distance(drone.coords, self.simulator.depot_coordinates)
			time_drone_to_depot = drone_depot_distance / drone.speed

			if drone.move_routing:
				if self.drone.move_routing:
					if drone.buffer_length() >= 1 and time_self_to_depot > time_drone_to_depot:
						tmp_score = drone.buffer_length() * self.q_table[cell_index][drone.identifier] / time_drone_to_depot
						if drone_score < tmp_score:
							action = drone
							drone_score = tmp_score							
				else:
					# I'm not going to the depot, while my neighbours are
					tmp_score = drone.buffer_length() * self.q_table[cell_index][drone.identifier] / time_drone_to_depot
					if drone_score < tmp_score:
						action = drone
						drone_score = tmp_score
			else:
				# Neither me nor my neighbours are going to the depot
				if not self.drone.move_routing and self.drone.buffer_length() < drone.buffer_length():
					action = drone
				# I'm going to the depot, while my neighbours are not
				elif self.drone.move_routing:
					action = None

		return action


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

		packet_id_event: int = pkd.event_ref.identifier
		neighbors_drones: Set[Drone] = {drone[1] for drone in opt_neighbors}

		if self.force_exploration or self.rnd_for_routing_ai.rand() < self.EPSILON:
			neighbors_drones.add(self.drone)
			action = self.__exploration(neighbors_drones)
		else:
			action = self.__exploitation_1795030(neighbors_drones, cell_index)
		

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
