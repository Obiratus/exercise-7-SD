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
    def solve(self, verbose=True):
        """
        Solves the TSP using the Ant System algorithm.

        This method:
        1. Loads the environment with TSP instance data
        2. Iteratively sends ants to construct tours
        3. Updates pheromones based on tour quality
        4. Tracks the best solution found

        Parameters:
        - verbose: If True, print iteration progress

        Returns:
        - The best tour found (list of cities)
        - The length of the best tour
        """
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
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Best distance = {shortest_distance}")

        return solution, shortest_distance

    # Update rho parameter and reset environment
    def update_rho(self, rho):
        """
        Updates the rho parameter and resets the environment with this new value.

        Parameters:
        - rho: new rho value (pheromone evaporation rate)
        """
        self.rho = rho

        # Reset environment with new rho value
        self.environment = Environment(self.rho)

        # Reset ants to join the new environment
        self.ants = []
        for i in range(self.ant_population):
            ant = Ant(self.alpha, self.beta, None)
            ant.join(self.environment)
            self.ants.append(ant)


def main():
    # Fixed parameters
    ant_population = 48
    iterations = 100
    alpha = 1.0
    beta = 5.0

    # Define test parameters for rho
    test_params = [
        {"id": "R1", "rho": 0.1, "description": "Very slow evaporation (pheromones persist long)"},
        {"id": "R2", "rho": 0.3, "description": "Mild evaporation"},
        {"id": "R3", "rho": 0.5, "description": "Balanced evaporation (reference baseline)"},
        {"id": "R4", "rho": 0.7, "description": "Fast evaporation"},
        {"id": "R5", "rho": 0.9, "description": "Very fast evaporation (pheromones decay quickly)"}
    ]

    # Initialize ant colony with baseline parameters
    ant_colony = AntColony(ant_population, iterations, alpha, beta, 0.5)

    # Run all tests
    test_results = []

    print(
        f"Running multiple tests with different rho values (fixed: population={ant_population}, iterations={iterations}, alpha={alpha}, beta={beta}):")
    print("-------------------------------------------------------------")

    for test in test_params:
        test_id = test["id"]
        rho = test["rho"]

        print(f"\nRunning {test_id}: ρ={rho} - {test['description']}")

        # Update rho for this test
        ant_colony.update_rho(rho)

        # Run the optimization
        solution, distance = ant_colony.solve(verbose=False)

        # Store results
        test_results.append({
            "id": test_id,
            "distance": distance,
            "tour": solution
        })

        # Print progress
        print(f"{test_id}: Distance {distance}")

    # Print the final summary
    print("\n-------------------------------------------------------------")
    print("TEST RESULTS SUMMARY")
    print("-------------------------------------------------------------")

    for result in test_results:
        tour_string = ','.join(str(city) for city in result["tour"])
        print(f"{result['id']}: Distance {result['distance']}, Tour: {tour_string}")


if __name__ == '__main__':
    main()