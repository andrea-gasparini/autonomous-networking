
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
from typing import List, Tuple, Dict, Union, Set, Literal
from src.entities.uav_entities import Drone, DataPacket
import math

Action = Union[Drone, Literal[-1], Literal[-2], None]

class GradientBanditAI(BASE_routing):

    def __init__(self, drone: Drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}
       
        self.H = np.zeros(simulator.n_drones + 2)
        self.alpha = 0.2
        self.decay = 0.8
        self.droprate = 10
        self.iterations = 0

        self.mean_reward = 0
        self.n = 0

    def feedback(self, drone: Drone, id_event, delay, outcome, depot_index = None):
        """ return a possible feedback, if the destination drone has received the packet """
        if id_event in self.taken_actions:

            if outcome == -1:
                action, prob_action, actions, mean_reward = self.taken_actions[id_event]
                reward = -2
                if action == -1 or action == -2:
                    self.H[action] += self.alpha * (reward - mean_reward) * (1 - prob_action[-1])
                else:
                    self.H[action.identifier] += self.alpha * (reward - mean_reward) * (1 - prob_action[actions.index(action)])


                for i, act in enumerate(actions):
                    if action == act:
                        continue

                    if act != -1 and act != -2:
                        self.H[act.identifier] -= self.alpha * (reward - mean_reward) * (prob_action[i])
                    else:
                        self.H[act] -= self.alpha * (reward - mean_reward) * (prob_action[-1])

                #not_taken_actions = np.array(actions) != action
                #self.H[not_taken_actions] -= self.alpha * (reward - mean_reward) * (prob_action[not_taken_actions])

            del self.taken_actions[id_event]

    def softmax(self, actions: List[Action]) -> None:
        if isinstance(actions[-1], int):
            new_H = [self.H[drone.identifier] for drone in actions[:-1]] + [self.H[actions[-1]]]
        else:
            new_H = [self.H[drone.identifier] for drone in actions]
        neighbors_H = np.array(new_H)
        self.prob_action = np.exp(neighbors_H - np.max(neighbors_H)) / np.sum(np.exp(neighbors_H - np.max(neighbors_H)), axis=0)
    
    def update_H(self, action, reward, actions):
        learning_rate = self.alpha * self.decay ** (math.floor((1 + self.iterations) / self.droprate))
        if action == -1 or action == -2:
            self.H[action] += learning_rate * (reward - self.mean_reward) * (1 - self.prob_action[action])
        else:
            self.H[action.identifier] += learning_rate * (reward - self.mean_reward) * (1 - self.prob_action[actions.index(action)])

        for i, act in enumerate(actions):
            if action == act:
                continue

            if act != -1 and act != -2:
                self.H[act.identifier] -= learning_rate * (reward - self.mean_reward) * (self.prob_action[i])
            else:
                self.H[act] -= learning_rate * (reward - self.mean_reward) * (self.prob_action[-1])

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
        #print(min_distance_list[min_distance_idx])
        if min_distance_idx >= 2 and min_distance_list[min_distance_idx] < 550:
            # min distance is first_depot_distance or second_depot_distance
            # move to the the nearest depot
            return -1 if first_depot_distance < second_depot_distance else -2
        else:
            # min distance is first_depot_distance_next or second_depot_distance_next
            # do not transmit the packet and move to the next target
            return self.drone

    def choose_by_drone_score(self, actions: List[Action], ) -> List[Action]:

        if len(actions) == 2:
            return actions
            
        first_depot_coordinates = self.simulator.depot.list_of_coords[0]
        me_to_first_depot = util.euclidean_distance(self.drone.coords, first_depot_coordinates)

        second_depot_coordinates = self.simulator.depot.list_of_coords[1]
        me_to_second_depot = util.euclidean_distance(self.drone.coords, second_depot_coordinates)


        my_distance = me_to_first_depot if me_to_first_depot < me_to_second_depot else me_to_second_depot
        my_score = self.drone.speed + (math.log(self.drone.buffer_length()) if self.drone.buffer_length() > 0 else 0) / (my_distance + 0.0001)

        drone_score = my_score
        actions_to_return = [self.drone]
        for action in actions[:-1]:
            action_to_first_depot = util.euclidean_distance(action.coords, first_depot_coordinates)
            action_to_second_depot = util.euclidean_distance(action.coords, second_depot_coordinates)
            distance = action_to_first_depot if action_to_first_depot < action_to_second_depot else action_to_second_depot
            tmp_drone_score = (action.speed + (math.log(action.buffer_length()) if action.buffer_length() > 0 else 0)) / (distance + 0.0001)

            if tmp_drone_score >= drone_score:
                actions_to_return.append(action)
        
        #if len(actions_to_return) == 1:
        ret = self.calculate_best_depot(self.drone.coords, self.drone.next_target(), [first_depot_coordinates, second_depot_coordinates])
        if ret not in actions_to_return:
            actions_to_return.append(ret)
        #else:
        #    actions_to_return.append(actions[-1])
            
        return actions_to_return
        
    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score -> geographical approach, take the drone closest to the depot """

        action = None
        self.iterations += 1
        first_depot_coordinates = self.simulator.depot.list_of_coords[0]
        me_to_first_depot = util.euclidean_distance(self.drone.coords, first_depot_coordinates)

        second_depot_coordinates = self.simulator.depot.list_of_coords[1]
        me_to_second_depot = util.euclidean_distance(self.drone.coords, second_depot_coordinates)

        #if not self.drone.move_routing:
        actions: List[Action] = list({drone[1] for drone in opt_neighbors})
        actions.append(self.drone)
        #else:
        #    actions: List[Action] = list({drone[1] for drone in opt_neighbors if drone[1].move_routing})
        #    actions.append(self.drone)
        
        # l = [sdrone, -2], l[-2] = sdrone

        #if me_to_f#irst_depot < me_to_second_depot:
        #    actions.append(-1)
        #else:
        #    actions.append(-2)

        actions = self.choose_by_drone_score(actions)
        self.softmax(actions)
        action = np.random.choice(actions, p=self.prob_action)
        
        avg_packets = 0
        for pkt in self.drone.all_packets():
            avg_packets += self.simulator.cur_step - pkt.time_step_creation
        avg_packets = avg_packets / self.drone.buffer_length()

        if action != -1 and action != -2:
            action_to_first_depot = util.euclidean_distance(action.coords, first_depot_coordinates)
            action_to_second_depot = util.euclidean_distance(action.coords, second_depot_coordinates)
            distance = action_to_first_depot if action_to_first_depot < action_to_second_depot else action_to_second_depot
            reward = action.speed * action.buffer_length() / distance
        else:
            distance = me_to_first_depot if me_to_first_depot < me_to_second_depot else me_to_second_depot
            reward = self.drone.speed * self.drone.buffer_length() / distance
        
            #self.simulator.metrics.energy_spent_for_active_movement[self.drone.identifier]    

        self.n += 1
        self.mean_reward += (reward - self.mean_reward) / self.n
        #print(reward - self.mean_reward, reward)

        self.update_H(action, reward, actions)

        self.taken_actions[pkd.event_ref.identifier] = action, self.prob_action, actions, self.mean_reward

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
