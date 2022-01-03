
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket
import math

Action = Union[Drone, Literal[-1], Literal[-2], None]

class AIThreeRouting(BASE_routing):

    BUFFER_LENGTH_THRESHOLD = 2

    def __init__(self, drone: Drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}
    
        self.H = np.zeros(simulator.n_drones + 2)
        self.alpha = 0.2
        self.decay = 0.6
        self.droprate = 10
        self.iterations = 0

        self.q_table: Dict[int, List[int]] = {} # {0: [0, ...., 0]}

    def calculate_lr(self):
        """ Calculate learning rate """
        return self.alpha * self.decay ** (math.floor((1 + self.iterations) / self.droprate))

    def feedback(self, drone: Drone, id_event, delay, outcome, depot_index = None):
        """ return a possible feedback, if the destination drone has received the packet """
        if id_event in self.taken_actions:

            action, cell_index, next_cell_index = self.taken_actions[id_event]

            if outcome == -1:
                if isinstance(action, Drone):
                    self.q_table[cell_index][action.identifier] += self.calculate_lr() * (-2 + 0.6 * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action.identifier])
                else:
                    self.q_table[cell_index][action] += self.calculate_lr() * (-2 + 0.6 * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action])
                
            del self.taken_actions[id_event]

    def get_nearest_depot(self, drone):
        first_depot_coordinates = self.simulator.depot.list_of_coords[0]
        me_to_first_depot = util.euclidean_distance(drone.coords, first_depot_coordinates)

        second_depot_coordinates = self.simulator.depot.list_of_coords[1]
        me_to_second_depot = util.euclidean_distance(drone.coords, second_depot_coordinates)

        return (-1, first_depot_coordinates) if me_to_first_depot < me_to_second_depot else (-2, second_depot_coordinates)

    def necessary_energy_to_coords(self, drone: Drone, coords: Tuple[int, int]) -> float: 
        return util.euclidean_distance(drone.coords, coords) / drone.speed

    def energy_spent_until_now(self, drone: Drone) -> float:
        return self.necessary_energy_to_coords(drone, drone.last_mission_coords)

    def drone_returning_to_depot(self, drones_to_depot):
        my_depot, my_depot_coords = self.get_nearest_depot(self.drone)

        drone_to_check = self.drone
        energy_spent = self.energy_spent_until_now(self.drone)
        energy_remaining_depot = self.necessary_energy_to_coords(self.drone, my_depot_coords)

        for drone in drones_to_depot:
            drone_depot, drone_depot_coords = self.get_nearest_depot(drone)
            drone_energy_spent = self.energy_spent_until_now(drone)
            drone_energy_remaining_to_depot = self.necessary_energy_to_coords(drone, drone_depot_coords)

            if drone_energy_spent + drone_energy_remaining_to_depot < energy_spent + energy_remaining_depot:
                drone_to_check = drone
                energy_spent = drone_energy_spent
                energy_remaining_depot = drone_energy_remaining_to_depot
            elif drone_energy_spent + drone_energy_remaining_to_depot == energy_spent + energy_remaining_depot:
                if drone.buffer_length() > drone_to_check.buffer_length():
                    drone_to_check = drone
                    energy_spent = drone_energy_spent
                    energy_remaining_depot = drone_energy_remaining_to_depot

        return drone_to_check
        # distanza dal punto in cui mi son girato per tornare al depot (self.drone.last_mission_coords)

    def drone_not_returning_to_depot_w_neigh(self, drones: List[Drone]):
        my_depot, my_depot_coords = self.get_nearest_depot(self.drone)

        drone_to_check = self.drone
        energy_remaining_to_depot = self.necessary_energy_to_coords(self.drone, my_depot_coords)
        for drone in drones:
            drone_depot, drone_depot_coords = self.get_nearest_depot(drone)
            drone_energy_spent = self.energy_spent_until_now(drone)
            drone_energy_remaining_to_depot = self.necessary_energy_to_coords(drone, drone_depot_coords)

            if energy_remaining_to_depot > drone_energy_spent + drone_energy_remaining_to_depot:
                drone_to_check = drone
                energy_remaining_to_depot = drone_energy_spent + drone_energy_remaining_to_depot

        return drone_to_check

    def get_drone_nearest_depot(self, drones):
        my_depot, my_depot_coords = self.get_nearest_depot(self.drone)
        my_distance = util.euclidean_distance(self.drone.coords, my_depot_coords)
        drone_to_check = self.drone

        for drone in drones:
            drone_depot, drone_depot_coords = self.get_nearest_depot(drone)
            drone_distance = util.euclidean_distance(drone.coords, drone_depot_coords)

            if drone_distance < my_distance:
                my_distance = drone_distance
                drone_to_check = drone

        avg_packets = self.calculate_avg_packets_time(self.drone)

        if drone_to_check == self.drone and avg_packets > 500:
            return my_depot, avg_packets
        
        return drone_to_check, avg_packets

    def calculate_avg_packets_time(self, drone):
        avg_packets = 0
        for pkt in drone.all_packets():
            avg_packets += self.simulator.cur_step - pkt.time_step_creation
        avg_packets /= (drone.buffer_length() + 0.001)

        return avg_packets + 0.0001

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score -> geographical approach, take the drone closest to the depot """

        action: Action = self.drone
        neighbors_drones: Set[Drone] = {drone[1] for drone in opt_neighbors}

        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                        width_area=self.simulator.env_width,
                                                        x_pos=self.drone.coords[0],  # e.g. 1500
                                                        y_pos=self.drone.coords[1])[0]  # e.g. 500

        if cell_index not in self.q_table:
            self.q_table[cell_index] = [0 for i in range(self.simulator.n_drones + len(self.simulator.depot.list_of_coords))]
            
        self.iterations += 1

        drones_returning_to_depot = [drone for drone in neighbors_drones if drone.move_routing]

        reward = 0

        if self.drone.buffer_length() >= self.BUFFER_LENGTH_THRESHOLD:
            if len(drones_returning_to_depot) > 0:
                if self.drone.move_routing:
                    action = self.drone_returning_to_depot(drones_returning_to_depot)
                    reward = 1
                else:
                    action = self.drone_not_returning_to_depot_w_neigh(drones_returning_to_depot)
                    avg_packets = self.calculate_avg_packets_time(self.drone)
                    reward = 1 / avg_packets * action.buffer_length()
            
            elif not self.drone.move_routing:
                if len(neighbors_drones) > 0:
                    action, avg_packets = self.get_drone_nearest_depot(neighbors_drones)
                    if action in [-1, -2]:
                        reward = 1 / avg_packets
                    else:
                        reward = 1 / avg_packets * action.buffer_length()
                else:
                    avg_packets = self.calculate_avg_packets_time(self.drone)
                    if avg_packets > 500:
                        my_depot, my_depot_coords = self.get_nearest_depot(self.drone)
                        action = my_depot
                    reward = 1 / avg_packets
            else: # self.drone.move_routing and not len(drones_returning_to_depot) > 0
                reward = 1
        else:
            if len(neighbors_drones) > 0:
                neighbors_drones_q_table = [self.q_table[cell_index][drone.identifier] for drone in neighbors_drones]
                max_value = max(neighbors_drones_q_table)
                if max_value > 0:
                    action = [drone for drone in neighbors_drones if self.q_table[cell_index][drone.identifier] == max_value][0]
                else:
                    action = self.rnd_for_routing_ai.choice(list(neighbors_drones))

                reward = 0.5
            else:
                action = self.drone
                avg_packets = self.calculate_avg_packets_time(self.drone)
                reward = 1 / avg_packets

        next_cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                            width_area=self.simulator.env_width,
                                                            x_pos=self.drone.next_target()[0],
                                                            y_pos=self.drone.next_target()[1])[0]
        
        if next_cell_index not in self.q_table:
            self.q_table[next_cell_index] = [0 for i in range(self.simulator.n_drones + len(self.simulator.depot.list_of_coords))]

        if isinstance(action, Drone):
            self.q_table[cell_index][action.identifier] += self.calculate_lr() * (reward + 0.6 * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action.identifier])
        else:
            self.q_table[cell_index][action] += self.calculate_lr() * (reward + 0.6 * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action])
            
        self.taken_actions[pkd.event_ref.identifier] = (action, cell_index, next_cell_index)

        return action

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        pass
