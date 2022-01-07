
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket

class TestAIRouting(BASE_routing):
    def __init__(self, drone: Drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_action)
        self.epsilon = 0.02
        self.q_table: Dict[int, List[int]] = {} # {0: [0, ...., 0]}
        self.force_exploration = True
        self.pkts_transmitted = {}

    def feedback(self, drone, id_event, delay, outcome, depot_index=None):
        """ return a possible feedback, if the destination drone has received the packet """
        if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:",  delay, " - outcome:", outcome,
                  " - to depot: ", depot_index)

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
        # NOTE: reward or update using the old action!!
        # STORE WHICH ACTION DID YOU TAKE IN THE PAST.
        # do something or train the model (?)
        if id_event in self.taken_actions:
            action, cell_index, next_cell_index, time = self.taken_actions[id_event]
            action_id = action
            if action == None:
                action_id = self.drone.identifier
                speed = self.drone.speed
            elif isinstance(action, Drone):
                action_id = action.identifier
                speed = action.speed
            else:
                speed = 1

            if outcome == -1:
                reward = -2
            else:
                reward = 2 * speed

            self.q_table[cell_index][action_id] += 0.67 * (reward + 0.8 * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action_id])
            
            del self.taken_actions[id_event]

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        # Notice all the drones have different speed, and radio performance!!
        # you know the speed, not the radio performance.
        # self.drone.speed

        # Only if you need --> several features:
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                        width_area=self.simulator.env_width,
                                                        x_pos=self.drone.coords[0],  # e.g. 1500
                                                        y_pos=self.drone.coords[1])[0]  # e.g. 500
        #print("Drone: ", self.drone.identifier, " - i-th cell:",  cell_index, " - center:", self.simulator.cell_to_center_coords[cell_index])

        if cell_index not in self.q_table:
            self.q_table[cell_index] = [0 for i in range(self.simulator.n_drones + len(self.simulator.depot.list_of_coords))]

        action = None

        # self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
        # self.drone.residual_energy (that tells us when I'll come back to the depot).
        #  .....
        #for hpk, drone_instance in opt_neighbors:
            #print(hpk)
        #    continue

        first_depot_coordinates = self.simulator.depot.list_of_coords[0]
        second_depot_coordinates = self.simulator.depot.list_of_coords[1]

        neighbor_drones: Set[Drone] = {drone[1] for drone in opt_neighbors}
        first_depot_distance_time = util.euclidean_distance(self.drone.coords, first_depot_coordinates) / self.drone.speed
        second_depot_distance_time = util.euclidean_distance(self.drone.coords, second_depot_coordinates) / self.drone.speed

        if self.force_exploration or self.rnd_for_routing_ai.rand() < self.epsilon:
            action = self.rnd_for_routing_ai.choice(list(neighbor_drones)) if len(neighbor_drones) > 0 else None
            self.force_exploration = False
        else:
            if len(neighbor_drones) == 0 and self.drone.buffer_length() > 0:
                action = -1 if first_depot_distance_time < second_depot_distance_time else -2
            else:
                max_value = max(self.q_table[cell_index][:self.simulator.n_drones])
                if max_value > 0:
                    best_action = self.q_table[cell_index].index(max(self.q_table[cell_index][:self.simulator.n_drones]))
                    drones_id: Set[int] = {drone.identifier for drone in neighbor_drones}
                    if best_action in drones_id:
                        action = [drone for drone in neighbor_drones if drone.identifier == best_action][0]
                else:
                    # I choice the best drone that is returning.
                    drones_returning = sorted([(drone, drone.speed) for drone in neighbor_drones if drone.move_routing], key=lambda x: x[1])
                    if len(drones_returning) > 0:
                        action = drones_returning[0][0]
                    else:
                        action = None if self.drone.buffer_length() <= 0 else -1 if first_depot_distance_time < second_depot_distance_time else -2

            
        next_cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                            width_area=self.simulator.env_width,
                                                            x_pos=self.drone.next_target()[0],
                                                            y_pos=self.drone.next_target()[1])[0]
        
        if next_cell_index not in self.q_table:
            self.q_table[next_cell_index] = [0 for i in range(self.simulator.n_drones + len(self.simulator.depot.list_of_coords))]
        # return action:
        # None --> no transmission
        # -1 --> move to first depot (self.simulator.depot.list_of_coords[0]
        # -2 --> move to second depot (self.simulator.depot.list_of_coords[1]
        # 0, ... , self.ndrones --> send packet to this drone
        return action  # here you should return a drone object!


    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        pass
