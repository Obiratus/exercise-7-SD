# Exercise 7: Interacting Agents on the Web
The git repository can be found here: https://github.com/Obiratus/exercise-7-SD

## Task 3: Working with Ant Colony Optimization

### Task 3.1: Impact of Parameters α and β on ACO Performance
Run the following code for the experiment:
```
ant-colony-task3.1.py
```
This experiment analyzes how parameters α and β affect the Ant Colony Optimization (ACO) algorithm performance. Tests were run on the ATT48 symmetric TSP instance with an optimal known length of 10628. For all tests, ant population (48 ants), iterations (100), and pheromone evaporation rate (ρ = 0.5) were constant.

The tests included various combinations of α and β values:

| Test ID | α (alpha) | β (beta) | Description                             |
|---------|-----------|----------|-----------------------------------------|
| T1      | 0.5       | 1.0      | Low pheromone, low heuristic            |
| T2      | 1.0       | 2.0      | Balanced baseline                       |
| T3      | 1.0       | 5.0      | Strong heuristic bias                   |
| T4      | 2.0       | 5.0      | Increased pheromone and heuristic       |
| T5      | 2.0       | 2.0      | Moderate equal influence                |
| T6      | 3.0       | 1.0      | High pheromone, low heuristic           |
| T7      | 5.0       | 1.0      | Strong pheromone bias                   |
| T8      | 0.5       | 5.0      | Low pheromone, high heuristic           |

#### Results Summary

| Test ID | α | β | Distance | Deviation from Optimal |
|---------|---|---|----------|-------------------------|
| T1      | 0.5 | 1.0 | 16487 | +5859 |
| T2      | 1.0 | 2.0 | 11024 | +396 |
| T3      | 1.0 | 5.0 | 11173 | +545 |
| T4      | 2.0 | 5.0 | 11618 | +990 |
| T5      | 2.0 | 2.0 | 12098 | +1470 |
| T6      | 3.0 | 1.0 | 12530 | +1902 |
| T7      | 5.0 | 1.0 | 12530 | +1902 |
| T8      | 0.5 | 5.0 | 12530 | +1902 |

#### Analysis and Interpretation

Parameters α and β determine the trade-off between exploitation (pheromone trails) and exploration (heuristic information):

- Balanced Parameters (T2: α=1.0, β=2.0):
    - Produced the best performance (11024), closest to the optimal (10628).
    - Indicates effective balance, enabling exploration and exploiting good routes.

- High Heuristic Bias (T3, T8):
    - T3 (α=1.0, β=5.0) had good performance (11173) showing effectiveness of heuristic guidance.
    - T8 (α=0.5, β=5.0) performed poorly (12530) due to insufficient pheromone reinforcement, limiting effective path reuse.

- High Pheromone Bias (T6, T7):
    - Both T6 (α=3.0, β=1.0) and T7 (α=5.0, β=1.0) stagnated at the same suboptimal solution (12530).
    - Excessive reliance on pheromone trails led ants to premature convergence on suboptimal paths.

- Weak Influence (T1: α=0.5, β=1.0):
    - Performed worst (16487), due to minimal guidance from either heuristic or pheromone information.

#### Conclusion

ACO performance strongly depends on appropriately balanced α and β values. Moderate values (α ≈ 1–2, β ≈ 2–5) yield the best results, while extremes in either direction significantly degrade performance due to over-exploitation or insufficient guidance.

### Task 3.2: Impact of Evaporation Rate ρ on ACO Performance
Run the following code for the experiment:
```
ant-colony-task3.2.py
```

This experiment evaluates the effect of the pheromone evaporation rate (ρ) on the performance of the Ant Colony Optimization algorithm. The experiment was conducted on the ATT48 TSP instance with an optimal solution length of 10628.

The following parameters were fixed:
- Ant population: 48
- Iterations: 100
- α (alpha): 1.0
  - best value as shown in experiment 3.1
- β (beta): 5.0
  - best value as shown in experiment 3.1

Five different ρ values were tested to observe how quickly pheromone trails decay over time.

| Test ID | ρ (rho) | Description                                 |
|---------|---------|---------------------------------------------|
| R1      | 0.1     | Very slow evaporation                      |
| R2      | 0.3     | Mild evaporation                           |
| R3      | 0.5     | Balanced evaporation (baseline)            |
| R4      | 0.7     | Fast evaporation                           |
| R5      | 0.9     | Very fast evaporation (rapid decay)        |

#### Results Summary

| Test ID | ρ | Distance | Deviation from Optimal |
|---------|---|----------|-------------------------|
| R1      | 0.1 | 11278   | +650                   |
| R2      | 0.3 | 11345   | +717                   |
| R3      | 0.5 | 11189   | +561                   |
| R4      | 0.7 | 11263   | +635                   |
| R5      | 0.9 | 11503   | +875                   |

#### Analysis and Interpretation

- Best result (R3) was achieved with ρ = 0.5, indicating that a balanced level of pheromone evaporation supports both learning and exploration. It reinforces good paths long enough while allowing outdated ones to fade.

- Low evaporation (R1, ρ = 0.1) produced relatively good performance, but the slower decay may preserve early, suboptimal paths too long, limiting the adaptability of the colony.

- Moderate evaporation (R2, ρ = 0.3) showed slightly weaker performance than ρ = 0.5, possibly due to a slower transition from exploration to exploitation.

- High evaporation (R4 and R5, ρ ≥ 0.7) led to weaker performance. As pheromones decayed quickly, ants had less consistent reinforcement and relied more on random exploration, making it harder to stabilize around high-quality paths.

- Very fast evaporation (R5) produced the worst result, suggesting that excessive pheromone loss prevents the colony from leveraging useful past experiences.

#### Conclusion

Evaporation rate plays a key role in the convergence behavior of ACO. A moderate value (around ρ = 0.5) gives the best performance by maintaining a dynamic balance between preserving useful information and promoting exploration. Extremely low or high values reduce effectiveness by either trapping the search or disrupting convergence.

### Task 3.3: Adapting ACO for a Dynamic Traveling Salesman Problem (DTSP)

The Dynamic TSP introduces runtime modifications to the set of cities. To handle this scenario using the current Ant Colony Optimization implementation, the following modifications are necessary:

### 1. Dynamic Environment Updates

- Real-time Node Management:
    - Implement methods in the `Environment` class to dynamically add and remove cities at runtime.
    - Upon addition or removal, recalculate distances and pheromone trails involving affected nodes immediately.

```python
def add_city(self, city_id, coordinates):
    self.coordinates[city_id] = coordinates
    # Update distances and pheromones for new city
    # Initialize pheromone values for new edges

def remove_city(self, city_id):
    del self.coordinates[city_id]
    # Remove all associated distances and pheromones
```

### 2. Flexible Pheromone Management

- Adaptive Initialization:
    - Initialize pheromone values for newly added cities/edges using the existing heuristic (e.g., nearest neighbor approximation).

- Dynamic Evaporation Adjustments:
    - Consider temporary pheromone resets or boosts in evaporation rates to handle sudden topological changes.
[.devcontainer](../.devcontainer)
### 3. Reactive Ant Behavior

- Dynamic Tours:
    - Ants must check the environment state at each iteration:
        - If cities have been added, ants should incorporate new nodes immediately.
        - If cities have been removed, ants should avoid removed nodes and recalculate their ongoing tours if necessary.

- Recovery Mechanism:
    - Implement a simple method to quickly reset ants' state after major changes, ensuring that ants construct valid tours based on the current node set.

```python
# Inside Ant class:
def validate_tour(self):
    # Remove invalid (deleted) cities from current tour
    # Add logic to redirect to valid city if necessary
```

### 4. Event-driven Approach

- Event Detection:
    - Implement an event-driven architecture where the ant colony is notified immediately whenever the environment topology changes.
    - Ant colony reacts by adjusting pheromone trails and ant tours dynamically.

### 5. Enhanced Robustness & Stability

- Adjust pheromone deposition rules temporarily after topology changes to encourage exploration, helping the colony rapidly adapt to new or missing cities.

### Conclusion

By adding dynamic node management, reactive ant behavior, adaptive pheromone strategies, and event-driven updates, the current ACO implementation can effectively handle dynamic changes inherent to DTSP problems.

## Declaration of aids

| Task   | Aid                   | Description                     |
|--------|-----------------------|---------------------------------|
| Task 1 | IntelliJ AI Assistant | Explain code, Help with syntax. |
| Task 2 | IntelliJ AI Assistant | Explain code, Help with syntax. |
| Task 3 | IntelliJ AI Assistant | Help with syntax. |
| Task 3 | Language tool         | Grammar and spell check.        |

