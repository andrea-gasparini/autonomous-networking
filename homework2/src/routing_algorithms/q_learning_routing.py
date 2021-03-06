import numpy as np
import json
import os

from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config

class QLearningRouting(BASE_routing):

	ALPHA: float = 0.6
	"""
	Learning rate
	"""
	GAMMA: float = 0.8
	"""
	Discount factor, i.e. the importance of future rewards
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
		self.tmp_energy = 0
		self.is_returning = False


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
			selected_drone = self.drone
			if action == None:
				#selected_drone = self.drone
				action = self.drone.identifier
			elif isinstance(action, Drone):
				selected_drone = action
				action = action.identifier
			
			if outcome == -1:
				reward = -2
			else:
				buffer_len = drone.buffer_length() if action != -1 else self.drone.buffer_length()
				reward = 2 * buffer_len
				if action == -1 and drone == self.drone:
					energy_spent = self.simulator.metrics.energy_spent_for_active_movement[self.drone.identifier]

					reward += 1 / energy_spent

			self.q_table[cell_index][action] += self.ALPHA * (reward + self.GAMMA * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action])

			del self.taken_actions[id_event]


	def __exploration(self, neighbors_drones: Set[Drone]) -> Union[None, Literal[-1], Drone]:
		""" Do exploration """
		action = None
		#send_to_rnd_drone: bool = self.rnd_for_routing_ai.rand() > 1 - self.EXPLORATION_SEND_PROBABILTY
		neighbors_returning_to_depot: List[Drone] = [drone for drone in neighbors_drones if drone.move_routing]
		neighbors_with_packets: List[Drone] = [drone for drone in neighbors_drones if self.drone.buffer_length() < drone.buffer_length()]

		if len(neighbors_drones) == 0:
			action = None
		elif 1 <= self.drone.buffer_length() < 4:
			action = None if len(neighbors_with_packets) == 0 else self.rnd_for_routing_ai.choice(list(neighbors_with_packets))
		elif self.drone.buffer_length() >= 4 and not self.drone.move_routing and len(neighbors_drones) > 0:
			action = -1 if len(neighbors_returning_to_depot) == 0 else self.rnd_for_routing_ai.choice(list(neighbors_returning_to_depot))
		else:
			action = None

		self.force_exploration = False

		return action if action != self.drone else None
		


	def __exploitation(self, neighbors_drones: Set[Drone], cell_index: int) -> Union[None, Literal[-1], Drone]:
		""" Do exploitation """	
		action = None
		self_depot_distance = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)
		if self_depot_distance <= 400:
			return -1
		
		if len(neighbors_drones) == 0 and self.drone.buffer_length() >= 2:
			return -1
		elif len(neighbors_drones) == 0:
			return None
		elif not self.drone.move_routing:
			max_score = float('-inf')
			for drone in neighbors_drones:
				if self.q_table[cell_index][drone.identifier] > max_score and self.drone.buffer_length() <= drone.buffer_length():
					action = drone
					max_score = self.q_table[cell_index][drone.identifier]
				
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

		neighbors_drones: Set[Drone] = {drone[1] for drone in opt_neighbors}

		if self.force_exploration or self.rnd_for_routing_ai.rand() < self.EPSILON:
			neighbors_drones.add(self.drone)
			action = self.__exploration(neighbors_drones)
		else:
			action = self.__exploitation(neighbors_drones, cell_index)
		
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

		assert os.path.isdir("data"), "Can not save metrics, data directory does not exist"

		with open(os.path.join("data", "metrics.json"), "a+") as f:

			obj = {
				"alpha": self.ALPHA,		## np.arange(0.4, 1, 0.2)
				"gamma": self.GAMMA,		## np.arange(0.4, 1, 0.2)
				"epsilon": self.EPSILON,	## np.arange(0.05, 0.21, 0.05)
				"exploration-send-probability": self.EXPLORATION_SEND_PROBABILTY,
				"n-drones": self.simulator.n_drones,
				"seed": self.simulator.seed,
				"score": self.simulator.score(),
			}

			print(obj)

			json.dump(obj, f)

			f.write("\n")
