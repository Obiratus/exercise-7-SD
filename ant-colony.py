import numpy as np

from environment import Environment
from ant import Ant 

# Class representing the ant colony
"""
    ant_population: the number of ants in the ant colony
    iterations: the number of iterations 
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
    rho: pheromone evaporation rate
"""
class AntColony:
    def __init__(self, ant_population: int, iterations: int, alpha: float, beta: float, rho: float):
        self.ant_population = ant_population
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho 

        # Initialize the environment of the ant colony
        self.environment = Environment(self.rho)

        # Initilize the list of ants of the ant colony
        self.ants = []

        # Initialize the ants of the ant colony
        for i in range(ant_population):
            
            # Initialize an ant on a random initial location 
            ant = Ant(self.alpha, self.beta, None)

            # Position the ant in the environment of the ant colony so that it can move around
            ant.join(self.environment)
        
            # Add the ant to the ant colony
            self.ants.append(ant)

    # Solve the ant colony optimization problem  
    def solve(self):
        """
        Solves the TSP using the Ant System algorithm.

        This method:
        1. Loads the environment with TSP instance data
        2. Iteratively sends ants to construct tours
        3. Updates pheromones based on tour quality
        4. Tracks the best solution found

        Returns:
        - The best tour found (list of cities)
        - The length of the best tour
        """
        # Load the environment with the TSP instance data
        # self.environment.load_from_file('att48.tsp')

        # Initialize random positions for ants
        all_locations = self.environment.get_possible_locations()

        # The solution will be a list of the visited cities
        solution = []

        # Initially, the shortest distance is set to infinite
        shortest_distance = np.inf

        # Initialize ants with random starting locations
        for ant in self.ants:
            # Assign a random starting location to each ant
            initial_location = np.random.choice(all_locations)
            ant.current_location = initial_location
            ant.visited_locations = {initial_location}
            ant.tour = [initial_location]
            ant.traveled_distance = 0

        # Run the algorithm for the specified number of iterations
        for iteration in range(self.iterations):
            # Each ant constructs a tour
            ant_tours = []
            ant_distances = []

            for ant in self.ants:
                # Let the ant construct a complete tour
                tour, distance = ant.run()
                ant_tours.append(tour)
                ant_distances.append(distance)

                # Update the best solution if a better one is found
                if distance < shortest_distance:
                    shortest_distance = distance
                    solution = tour.copy()

            # Update pheromones using the existing method in Environment class
            self.environment.update_pheromone_map(ant_tours, ant_distances)


            # Optional: Print progress information
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best distance = {shortest_distance}")

        return solution, shortest_distance


def main():
    # Intialize the ant colony
    # Parameters based on literature: ant_population, iterations, alpha, beta, rho
    ant_colony = AntColony(48, 100, 1.0, 5.0, 0.5)


    # Solve the ant colony optimization problem
    solution, distance = ant_colony.solve()
    print("\nSOLUTION")
    print(f"Distance:  {distance}")
    print("Tour")
    for city in solution:
        print(city)



if __name__ == '__main__':
    main()    