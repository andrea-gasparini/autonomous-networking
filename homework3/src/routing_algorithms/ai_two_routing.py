
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket
import math

class AiTwoRouting(BASE_routing):

    EPSILON = 0.02

    AVG_PCKT_THRESHOLD = 500

    TOTAL_TIME_AVG_PCKT_THRESHOLD = 1000

    def __init__(self, drone: Drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_action)
        self.q_table: Dict[int, List[int]] = {} # {0: [0, ...., 0]}
        self.force_exploration = True
        self.pkts_transmitted = {}
        self.max_value_count = 0
        self.max_q_table_count = 0
        self.neighbors_0 = 0
        self.exploration_count = 0

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
            action, cell_index, next_cell_index = self.taken_actions[id_event]
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
                reward = 2 / delay

            self.q_table[cell_index][action_id] += 0.4 * (reward + 0.6 * (max(self.q_table[next_cell_index])) - self.q_table[cell_index][action_id])
            
            del self.taken_actions[id_event]

    def calculate_best_depot(self, coords, next_target, depots_coords):
        first_depot_coordinates = depots_coords[0]
        second_depot_coordinates = depots_coords[1]

        first_depot_distance = util.euclidean_distance(coords, first_depot_coordinates)
        second_depot_distance = util.euclidean_distance(coords, second_depot_coordinates)

        first_depot_distance_next = util.euclidean_distance(next_target, first_depot_coordinates)
        second_depot_distance_next = util.euclidean_distance(next_target, second_depot_coordinates)

        min_distance_list = [
            first_depot_distance_next,
            second_depot_distance_next,
            first_depot_distance,
            second_depot_distance
        ]

        min_distance_idx = np.argmin(min_distance_list)

        if min_distance_idx >= 2 and min_distance_list[min_distance_idx] < 550 and min_distance_list[min_distance_idx] > 200:
            # min distance is first_depot_distance or second_depot_distance
            # move to the the nearest depot
            return -1 if first_depot_distance < second_depot_distance else -2
        else:
            # min distance is first_depot_distance_next or second_depot_distance_next
            # do not transmit the packet and move to the next target
            return None

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


        neighbors_drones: Set[Drone] = {drone[1] for drone in opt_neighbors}

        first_depot_coordinates = self.simulator.depot.list_of_coords[0]
        second_depot_coordinates = self.simulator.depot.list_of_coords[1]

        if self.force_exploration or self.rnd_for_routing_ai.rand() < self.EPSILON:
            self.exploration_count += 1
            action = None if len(neighbors_drones) == 0 else self.rnd_for_routing_ai.choice(list(neighbors_drones))
            self.force_exploration = False
        else:
            first_depot_distance = util.euclidean_distance(self.drone.coords, first_depot_coordinates)
            second_depot_distance = util.euclidean_distance(self.drone.coords, second_depot_coordinates)

            first_depot_distance_next = util.euclidean_distance(self.drone.next_target(), first_depot_coordinates)
            second_depot_distance_next = util.euclidean_distance(self.drone.next_target(), second_depot_coordinates)

            next_target_distance = util.euclidean_distance(self.drone.next_target(), self.drone.coords)
            next_target_distance_time = next_target_distance / self.drone.speed

            first_depot_distance_time = first_depot_distance / self.drone.speed
            second_depot_distance_time = second_depot_distance / self.drone.speed

            first_depot_distance_time_next = first_depot_distance_next / self.drone.speed
            second_depot_distance_time_next = second_depot_distance_next / self.drone.speed
            

            if len(neighbors_drones) == 0 and self.drone.buffer_length() > 0:
                # mean time packets
                self.neighbors_0 += 1
                avg_packets = 0
                for pkt in self.drone.all_packets():
                    avg_packets += self.simulator.cur_step - pkt.time_step_creation
                avg_packets = avg_packets // self.drone.buffer_length()
                if avg_packets < self.AVG_PCKT_THRESHOLD:
                    next_target_depot = min(first_depot_distance_time_next, second_depot_distance_time_next)
                    total_time = next_target_depot + next_target_distance_time
                    if avg_packets + total_time < self.TOTAL_TIME_AVG_PCKT_THRESHOLD:
                        action = None
                    else:
                        action = self.calculate_best_depot(self.drone.coords, self.drone.next_target(), [first_depot_coordinates, second_depot_coordinates])
                        #if first_depot_distance_time < second_depot_distance_time:
                        #    action = -1
                        #else:
                        #    action = -2
                else:
                    action = self.calculate_best_depot(self.drone.coords, self.drone.next_target(), [first_depot_coordinates, second_depot_coordinates])
                    #if first_depot_distance_time < second_depot_distance_time:
                    #    action = -1
                    #else:
                    #    action = -2
            else:
                # move routing, speed, buffer length, distanza dai depot, next target
                neighbors_drones_q_table = [self.q_table[cell_index][drone.identifier] for drone in neighbors_drones]
                max_value = max(neighbors_drones_q_table)
                if max_value == 0:
                    self.max_value_count += 1
                    my_score = self.drone.speed + (math.log(self.drone.buffer_length()) if self.drone.buffer_length() > 0 else 0)
                    drone_score = my_score
                    
                    for drone in neighbors_drones:
                        tmp_drone_score = drone.speed + (math.log(drone.buffer_length()) if drone.buffer_length() > 0 else 0)
                        tmp_my_score = my_score
                        if not drone.move_routing:
                            drone_first_depot_distance = util.euclidean_distance(drone.next_target(), first_depot_coordinates)
                            drone_second_depot_distance = util.euclidean_distance(drone.next_target(), second_depot_coordinates)
                            #if drone_first_depot_distance < drone_second_depot_distance:
                            #    tmp_drone_score /= drone_first_depot_distance
                            #else:
                            #    tmp_drone_score /= drone_second_depot_distance
                        else:
                            drone_first_depot_distance = util.euclidean_distance(drone.coords, first_depot_coordinates)
                            drone_second_depot_distance = util.euclidean_distance(drone.coords, second_depot_coordinates)

                        if drone_first_depot_distance < drone_second_depot_distance:
                            tmp_drone_score /= drone_first_depot_distance
                            tmp_my_score /= first_depot_distance + 0.01 if drone.move_routing else first_depot_distance_next + 0.01 
                        else:
                            tmp_drone_score /= drone_second_depot_distance
                            tmp_my_score /= second_depot_distance + 0.01 if drone.move_routing else second_depot_distance_next + 0.01 
                        
                        #if self.drone.identifier == 8 and self.drone.move_routing and drone.identifier == 0:
                        #    print(self.drone, drone, tmp_drone_score, tmp_my_score, self.drone.speed, first_depot_distance_next)
                            
                        if drone.move_routing:
                            if tmp_drone_score > tmp_my_score:
                                drone_score = tmp_drone_score
                                action = drone 
                            
                        if tmp_drone_score >= drone_score:
                            drone_score = tmp_drone_score
                            action = drone
                    
                    if my_score == drone_score:
                        action = self.calculate_best_depot(self.drone.coords, self.drone.next_target(), [first_depot_coordinates, second_depot_coordinates])
                        #if first_depot_distance < second_depot_distance:
                        #    action = -1
                        #else:
                        #    action = -2
                else:
                    self.max_q_table_count += 1
                    action = [drone for drone in neighbors_drones if self.q_table[cell_index][drone.identifier] == max_value][0]
                        
        next_cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                            width_area=self.simulator.env_width,
                                                            x_pos=self.drone.next_target()[0],
                                                            y_pos=self.drone.next_target()[1])[0]
        
        if next_cell_index not in self.q_table:
            self.q_table[next_cell_index] = [0 for i in range(self.simulator.n_drones + len(self.simulator.depot.list_of_coords))]

        self.taken_actions[pkd.event_ref.identifier] = (action, cell_index, next_cell_index)
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
        print("max value count", self.max_value_count)
        print("max q table", self.max_q_table_count)
        print("neighbors 0 ", self.neighbors_0)
        print("Max Exploration count", self.exploration_count)

        import os
        import json

        assert os.path.isdir("data"), "Can not save metrics, data directory does not exist"

        with open(os.path.join("data", "metrics.json"), "a+") as f:

            obj = {
                "epsilon": self.EPSILON,		                                    ## list(np.arange(0.02, 0.8, 0.3)) + [0.8]
                "avg_pck_threshold": self.AVG_PCKT_THRESHOLD,                       ## np.arange(400, 600.1, 100)
                "total_time_avg_pck_threshold": self.TOTAL_TIME_AVG_PCKT_THRESHOLD, ## np.arange(800, 1200.1, 200)
                "n-drones": self.simulator.n_drones,
                "seed": self.simulator.seed,
                "score": self.simulator.score(),
                "sweep_path": config.SWEEP_PATH
            }

            print(obj)

            json.dump(obj, f)

            f.write("\n")