
import numpy as np
from typing import Dict, List, Set
from src.entities.uav_entities import DataPacket, Drone
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

class AIRouting(BASE_routing):
	def __init__(self, drone: Drone, simulator):
		BASE_routing.__init__(self, drone, simulator)
		# random generator
		self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
		self.taken_actions : Dict = dict()  #id event : (old_action)
		self.q_table : Dict = dict() # Q-table: {key: value}, key is a cell index, value a drone
		self.taken_actions_energy : Dict = dict() # id event : {cell_index: energy}
		self.time_step : int = 0 # an agent (drone) interacts with the environment in discrete time steps.
		self.epsilon : float = 0.98 # for epsilon-greedy (higher favors exploitation) 

	def _get_cell_index_for_feedback(self, drone : Drone, id_event : int) -> int:

		cells_in_id_event = list(self.taken_actions[id_event].keys())

		# if len(cells_in_id_event) == 1:
		#	return cells_in_id_event[0]
		
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


	def feedback(self, drone: Drone, id_event : int, delay : int, outcome) -> None:
		""" return a possible feedback, if the destination drone has received the packet """
		# Packets that we delivered and still need a feedback
		#print(self.drone.identifier, "----------", self.taken_actions)

		# outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
		# Feedback from a delivered or expired packet
		#print(self.drone.identifier, "----------", drone, id_event, delay, outcome)

		if drone == self.drone: # skip the feedback if we are the same drone.
			return None

		print(self._get_cell_index_for_feedback(drone, id_event))
		
			

		# Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
		# NOTE: reward or update using the old action!!
		# STORE WHICH ACTION DID YOU TAKE IN THE PAST.
		# do something or train the model (?)
		if id_event in self.taken_actions:
			action = self.taken_actions[id_event]
			del self.taken_actions[id_event]

	def _save_action(self, action : Drone, id_event : int = 0, cell_index : int = 0) -> None:
		if id_event not in self.taken_actions:
			self.taken_actions[id_event] = {cell_index: {action}} # create the key for this id_event with this cell and this drone
		else:
			if cell_index not in self.taken_actions[id_event]:
				self.taken_actions[id_event][cell_index] = {action} # the id event was in the taken actions but not on this cell, so add the cell
			else:
				self.taken_actions[id_event][cell_index].add(action) # otherwise, we just add the new Drone selected
		

	def _exploration(self, neighbors: List[Drone] = [], id_event : int = 0, cell_index : int = 0) -> Drone:
		""" Choose an action for exploration """
		neighbors_drones : Set[Drone] = {drone[1] for drone in neighbors} # Set of neighbors Drone objects
		action : Drone = self.rnd_for_routing_ai.choice(list(neighbors_drones)) # we take a drone randomly, this is exploration
		self._save_action(action=action, id_event=id_event, cell_index=cell_index)
		return action
		

	def _exploitation(self) -> Drone:
		pass

	def relay_selection(self, opt_neighbors: List[Drone], pkd: DataPacket):
		""" arg min score  -> geographical approach, take the drone closest to the depot """
		
		self.time_step += 1 # we are gonna to do an action, so increment the time-step
		
		packet_id_event : int = pkd.event_ref.identifier

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
		

		if not self.q_table[cell_index]:
			# we do exploration if the q_table[cell_index] is empty or with the e-greedy strategy
			return self._exploration(neighbors=opt_neighbors, id_event=packet_id_event, cell_index=cell_index)
		else:
			self._exploitation()

		action = None

		# self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
		# self.drone.residual_energy (that tells us when I'll come back to the depot).
		#  .....

		# Store your current action --- you can add several stuff if needed to take a reward later
		#self.taken_actions[pkd.event_ref.identifier] = (action)
		return None  # here you should return a drone object!

	def print(self):
		"""
			This method is called at the end of the simulation, can be usefull to print some
				metrics about the learning process
		"""
		pass
