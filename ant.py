import random
import math
import numpy as np

# Class representing an artificial ant of the ant colony
"""
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
"""


class Ant():
    def __init__(self, alpha: float, beta: float, initial_location):
        self.alpha = alpha
        self.beta = beta
        self.current_location = initial_location
        self.traveled_distance = 0
        self.tour = [initial_location]  # Track the tour
        self.visited_locations = {initial_location}  # For efficient lookup
        self.environment = None

    # The ant runs to visit all the possible locations of the environment
    def run(self):
        """
        Enables an ant to construct a complete tour by visiting all cities
        in the environment exactly once and returning to the starting city.

        This method:
        1. Resets the ant's state if it has already completed a tour
        2. Uses select_path() to choose cities based on pheromone and heuristic information
        3. Visits each city and keeps track of the tour and distance
        4. Returns to the starting city to complete the tour

        Returns:
        - The complete tour (list of locations)
        - The total distance traveled
        """
        # If ant has already completed a tour, reset its state
        if len(self.visited_locations) > 1:
            start_location = self.tour[0]
            self.tour = [start_location]
            self.visited_locations = {start_location}
            self.current_location = start_location
            self.traveled_distance = 0

        # Visit all cities
        all_locations = self.environment.get_possible_locations()

        # Continue until all locations have been visited
        while len(self.visited_locations) < len(all_locations):
            # Select next location to visit using the ant system rule
            next_location = self.select_path()

            # Visit the selected location
            self.visit(next_location)

        # Complete the tour by returning to the starting point
        # Only add the return journey if we're not already at the starting point
        if self.current_location != self.tour[0]:
            self.traveled_distance += self.get_distance(self.current_location, self.tour[0])

        return self.tour, self.traveled_distance

    # Select the next path based on the random proportional rule of the ACO algorithm
    def select_path(self):
        """
        Implements the random proportional rule (also called state transition rule)
        from the Ant System algorithm to select the next city to visit.

        The probability of moving from city i to city j is:

        p_ij = [τ_ij^α * η_ij^β] / Σ [τ_ik^α * η_ik^β]

        where:
        - τ_ij is the pheromone on edge (i,j)
        - η_ij is the heuristic value (1/distance) for edge (i,j)
        - α controls the influence of pheromone
        - β controls the influence of heuristic information
        - k ranges over all unvisited cities

        Returns the next location to visit
        """
        if len(self.visited_locations) == len(self.environment.get_possible_locations()):
            # All locations have been visited, return to the starting location
            return self.tour[0]

        # Get pheromone map from environment
        pheromone_map = self.environment.get_pheromone_map()

        # Calculate probabilities for all unvisited locations
        probabilities = {}
        denominator = 0.0

        # Get all possible next locations (that haven't been visited yet)
        for next_location in self.environment.get_possible_locations():
            if next_location not in self.visited_locations:
                # Get pheromone level on this edge
                pheromone = pheromone_map.get((self.current_location, next_location), 0.0)

                # Get distance to this location
                distance = self.get_distance(self.current_location, next_location)

                # Calculate heuristic value (inverse of distance)
                if distance > 0:
                    heuristic = 1.0 / distance
                else:
                    heuristic = 1.0  # Avoid division by zero

                # Calculate numerator: τ^α * η^β
                probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities[next_location] = probability
                denominator += probability

        # Normalize probabilities
        if denominator > 0:
            for location in probabilities:
                probabilities[location] /= denominator

        # Select next location based on probabilities
        # Using numpy's random.choice for weighted selection
        locations = list(probabilities.keys())
        weights = list(probabilities.values())

        # Check for edge case where all weights are zero
        if sum(weights) == 0:
            # If all weights are zero, choose randomly
            next_location = random.choice(locations)
        else:
            next_location = np.random.choice(locations, p=weights)

        return next_location

    # Position an ant in an environment
    def join(self, environment):
        self.environment = environment

    def get_distance(self, from_location, to_location):
        """
        Computes the pseudo-Euclidean distance between two cities
        as defined in the ATT distance metric of TSPLIB.

        The distance is calculated as:
        1. Calculate Euclidean distance and divide by 10.0
        2. Take the square root
        3. Round to the nearest integer
        4. If the rounded value is less than the actual value, add 1

        Parameters:
        - from_location: starting city
        - to_location: destination city

        Returns the distance between the two cities
        """
        # Get the distance directly from the environment
        # The environment already has the distances calculated according to the ATT metric
        return self.environment.get_distance(from_location, to_location)

    # Get the complete tour
    def get_tour(self):
        return self.tour

    # Get the tour length
    def get_tour_length(self):
        return self.traveled_distance

    # Add a location to the tour
    def visit(self, location):
        if location != self.current_location:
            # Add distance to traveled distance
            self.traveled_distance += self.get_distance(self.current_location, location)

            # Update current location
            self.current_location = location

            # Add to tour and visited locations
            self.tour.append(location)
            self.visited_locations.add(location)

            return True
        return False