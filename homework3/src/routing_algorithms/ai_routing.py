
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket

class AIRouting(BASE_routing):
    def __init__(self, drone: Drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_action)
        self.epsilon = 0.06
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
                reward = 2 / (time + 0.0001) * speed / delay

            self.q_table[cell_index][action_id] += 0.3 * (reward + 0.5 * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action_id])
            
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

        sum_steps = 0
        for packet in self.drone.all_packets():
            sum_steps += self.simulator.cur_step - packet.time_step_creation
        
        

        if cell_index not in self.q_table:
            self.q_table[cell_index] = [0 for i in range(self.simulator.n_drones + len(self.simulator.depot.list_of_coords))]

        if self.force_exploration or self.rnd_for_routing_ai.rand() < self.epsilon:
            if self.rnd_for_routing_ai.rand() < 0.5:
                if first_depot_distance_time < second_depot_distance_time and self.drone.buffer_length() >= 2:
                    action = -1
                elif first_depot_distance_time > second_depot_distance_time and self.drone.buffer_length() >= 2:
                    action = -2
            else:
                action = self.rnd_for_routing_ai.choice(list(neighbor_drones)) if len(neighbor_drones) > 0 else None
            
            self.force_exploration = False
        else:
            # exploitation
            
            drones_returning = [drone for drone in neighbor_drones if drone.move_routing]
            
            best_action = self.q_table[cell_index].index(max(self.q_table[cell_index]))
            drones_id = [drone.identifier for drone in neighbor_drones]
            drone_first_depot_distance = util.euclidean_distance(self.drone.next_target(), first_depot_coordinates)
            drone_second_depot_distance = util.euclidean_distance(self.drone.next_target(), second_depot_coordinates)
            avg_dist = (drone_first_depot_distance + drone_second_depot_distance) // 2

            if sum_steps // len(self.drone.all_packets()) >= 150 and len(neighbor_drones) == 0:
                action = -1 if first_depot_distance_time < second_depot_distance_time else -2
            elif best_action not in drones_id and not self.drone.move_routing:
                tmp_score = float('-inf')
                tmp_action = self.drone
                for drone in drones_returning:
                    drone_first_depot_distance = util.euclidean_distance(drone.next_target(), first_depot_coordinates)
                    drone_second_depot_distance = util.euclidean_distance(drone.next_target(), second_depot_coordinates)
                    avg_dist = (drone_first_depot_distance + drone_second_depot_distance) // 2
                    drone_score = drone.speed * drone.buffer_length() / avg_dist
                    if isinstance(tmp_action, Drone):
                        if tmp_score < drone_score:
                            tmp_action = drone
                            tmp_score = drone_score
                action = tmp_action        
            elif not self.drone.move_routing and best_action in drones_id:
                action = [drone for drone in neighbor_drones if drone.identifier == best_action][0]
            else:
                neighbor_drones.add(self.drone)
                best_drone_list = sorted([(drone, drone.speed) for drone in neighbor_drones if drone.move_routing], key=lambda x: (x[1]))
                if len(best_drone_list) <= 0:
                    action = None
                else:
                    action = best_drone_list[0][0]

        next_cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                            width_area=self.simulator.env_width,
                                                            x_pos=self.drone.next_target()[0],
                                                            y_pos=self.drone.next_target()[1])[0]
        
        if next_cell_index not in self.q_table:
            self.q_table[next_cell_index] = [0 for i in range(self.simulator.n_drones + len(self.simulator.depot.list_of_coords))]

        # Store your current action --- you can add several stuff if needed to take a reward later
        if action == -1 or action == -2:
            # we're returning to depot.. store also the time.
            if first_depot_distance_time < second_depot_distance_time:
                time = first_depot_distance_time
            else:
                time = second_depot_distance_time

            self.taken_actions[pkd.event_ref.identifier] = (action, cell_index, next_cell_index, time)
        else:
            if action == None:
                tm_drone = self.drone
            else:
                tm_drone = action
            drone_first_depot_distance = util.euclidean_distance(tm_drone.next_target(), first_depot_coordinates)
            drone_second_depot_distance = util.euclidean_distance(tm_drone.next_target(), second_depot_coordinates)

            sum_steps = 0
            for packet in tm_drone.all_packets():
                sum_steps += self.simulator.cur_step - packet.time_step_creation

            if drone_first_depot_distance < drone_second_depot_distance:
                time = drone_first_depot_distance / tm_drone.speed / (sum_steps // (len(tm_drone.all_packets()) + 1) + 1)
            else:
                time = drone_second_depot_distance / tm_drone.speed / (sum_steps // (len(tm_drone.all_packets()) + 1) + 1)

            self.taken_actions[pkd.event_ref.identifier] = (action, cell_index, next_cell_index, time)

            

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
