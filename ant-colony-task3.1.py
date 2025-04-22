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

    # Update alpha and beta parameters and reset ants
    def update_parameters(self, alpha, beta):
        """
        Updates the alpha and beta parameters and resets the ants with these new values.

        Parameters:
        - alpha: new alpha value
        - beta: new beta value
        """
        self.alpha = alpha
        self.beta = beta

        # Reset ants with new parameters
        self.ants = []
        for i in range(self.ant_population):
            ant = Ant(self.alpha, self.beta, None)
            ant.join(self.environment)
            self.ants.append(ant)


def main():
    # Define test parameters
    test_params = [
        {"id": "T1", "alpha": 0.5, "beta": 1.0, "description": "Very weak pheromone, low heuristic"},
        {"id": "T2", "alpha": 1.0, "beta": 2.0, "description": "Balanced (commonly used baseline)"},
        {"id": "T3", "alpha": 1.0, "beta": 5.0, "description": "Strong heuristic (original config)"},
        {"id": "T4", "alpha": 2.0, "beta": 5.0, "description": "Pheromone and heuristic boosted"},
        {"id": "T5", "alpha": 2.0, "beta": 2.0, "description": "Both influences equal and moderate"},
        {"id": "T6", "alpha": 3.0, "beta": 1.0, "description": "Strong pheromone, weak heuristic"},
        {"id": "T7", "alpha": 5.0, "beta": 1.0, "description": "Overly pheromone-biased (risk of stagnation)"},
        {"id": "T8", "alpha": 0.5, "beta": 5.0, "description": "Weak pheromone, very heuristic-heavy"}
    ]

    # Initialize ant colony with default parameters (will be updated for each test)
    ant_colony = AntColony(48, 100, 1.0, 5.0, 0.5)

    # Run all tests
    test_results = []

    print("Running multiple tests with different alpha and beta values:")
    print("-------------------------------------------------------------")

    for test in test_params:
        test_id = test["id"]
        alpha = test["alpha"]
        beta = test["beta"]

        print(f"\nRunning {test_id}: α={alpha}, β={beta} - {test['description']}")

        # Update parameters for this test
        ant_colony.update_parameters(alpha, beta)

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