import math
import tsplib95
import numpy as np

# Class representing the environment of the ant colony
"""
    rho: pheromone evaporation rate
"""


class Environment:
    def __init__(self, rho):
        self.rho = rho

        # Initialize the environment topology
        problem = tsplib95.load('att48-specs/att48.tsp')

        # Extract nodes and coordinates
        self.nodes = list(problem.get_nodes())
        self.coordinates = {}
        for node in self.nodes:
            self.coordinates[node] = problem.node_coords[node]

        # Calculate distances between cities using ATT distance metric
        self.distances = {}
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    x1, y1 = self.coordinates[i]
                    x2, y2 = self.coordinates[j]

                    # Calculate pseudo-Euclidean distance as per ATT standard
                    xd = x1 - x2
                    yd = y1 - y2
                    rij = math.sqrt((xd ** 2 + yd ** 2) / 10.0)
                    tij = int(round(rij))

                    # Adjust if tij < rij as per ATT standard
                    if tij < rij:
                        tij += 1

                    self.distances[(i, j)] = tij

        # Initialize the pheromone map in the environment
        self.pheromone_map = {}
        self.initialize_pheromone_map()

    # Initialize the pheromone trails in the environment
    def initialize_pheromone_map(self):
        """
        Initialize all pheromone trails with a small positive value.
        According to Dorigo & Stützle, a good initial value is τ0 = 1/(n*Lnn)
        where n is the number of cities and Lnn is the length of a nearest neighbor tour.
        """
        n = len(self.nodes)

        # Compute an approximate nearest neighbor tour length
        start_node = self.nodes[0]
        current_node = start_node
        unvisited = set(self.nodes)
        unvisited.remove(start_node)
        tour_length = 0

        # Build nearest neighbor tour
        while unvisited:
            next_node = min(unvisited, key=lambda node: self.distances.get((current_node, node), float('inf')))
            tour_length += self.distances[(current_node, next_node)]
            current_node = next_node
            unvisited.remove(current_node)

        # Complete the tour by returning to start
        tour_length += self.distances[(current_node, start_node)]

        # Calculate initial pheromone value
        initial_pheromone = 1.0 / (n * tour_length)

        # Initialize all edges with the same pheromone value
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    self.pheromone_map[(i, j)] = initial_pheromone

    # Update the pheromone trails in the environment
    def update_pheromone_map(self, ant_tours, ant_tour_lengths):
        """
        Update pheromone trails according to the Ant System algorithm:
        1. Evaporate pheromone on all edges
        2. Deposit new pheromone based on ants' tours

        Parameters:
        - ant_tours: List of tours (lists of nodes) for each ant
        - ant_tour_lengths: List of tour lengths corresponding to each ant's tour
        """
        # Evaporation phase: reduce pheromone on all edges
        for edge in self.pheromone_map:
            self.pheromone_map[edge] = (1 - self.rho) * self.pheromone_map[edge]

        # Deposit phase: add new pheromone based on ant tours
        for tour, tour_length in zip(ant_tours, ant_tour_lengths):
            # Calculate delta_tau for this ant (inverse of tour length)
            delta_tau = 1.0 / tour_length

            # Add pheromone on each edge of the tour
            for i in range(len(tour)):
                from_node = tour[i]
                to_node = tour[(i + 1) % len(tour)]  # Wrap around for the last city

                # Add pheromone to both directions (symmetric TSP)
                self.pheromone_map[(from_node, to_node)] += delta_tau
                self.pheromone_map[(to_node, from_node)] += delta_tau

    # Get the pheromone trails in the environment
    def get_pheromone_map(self):
        return self.pheromone_map

    # Get the environment topology (all possible locations)
    def get_possible_locations(self):
        return self.nodes

    # Get the distance between two locations
    def get_distance(self, from_node, to_node):
        return self.distances.get((from_node, to_node), float('inf'))