# Exercise 7: Stigmergic interaction for Smart Colony Optimization
Command for PDF
pandoc WAS_Simon_Dudler_Exercise7.md -o WAS_Simon_Dudler_Exercise7.pdf --pdf-engine=xelatex -V geometry:margin=0.75in

This repository contains a partial implementation of an Ant System algorithm \[[1],[2]\] for solving the [att48](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/) Travelling Salesman Problem instance. 

- \[[1]\] Dorigo, M., & Stützle, T. (2004). Ant colony optimization. The MIT Press.
- \[[2]\] M. Dorigo, M. Birattari, and T. Stutzle, “Ant colony optimization,” IEEE Computational Intelligence Magazine, vol. 1, no. 4, pp. 28–39, Nov. 2006, doi: 10.1109/mci.2006.329691

[1]: <https://dl.icdst.org/pdfs/files4/237b1d0e920bc5d59b75462e283d1559.pdf> "Ant colony optimization"
[2]: <https://www.researchgate.net/publication/308953674_Ant_Colony_Optimization> "Ant colony optimization"

## Table of Contents
- [Project structure](#project-structure)
- [How to run the project](#how-to-run-the-project)
- [Task 1](#task-1)
  - [Task 1.1](#task-11)
  - [Task 1.2](#task-12)
- [Task 2](#task-2)
  - [Task 2.1](#task-21)
  - [Task 2.2](#task-22)
  - [Task 2.3](#task-23)
- [Documentation](#documentation)

## Project structure
```bash
├── ant-colony.py # the ant colony that behaves base on the Ant System algorithm
├── ant.py # an artificial ant of the ant colony
├── environment.py # the environment of the ant colony
└── att48-specs # specification of the att48 TSP
    ├── att48.tsp # the specification of the TSP instance
    ├── att48.opt.tour # the optimal solution to the TSP instance
    ├── att48.opt.png 
    ├── att48_coordinates.txt 
    └── att48_distance_matrix.txt
```

## How to run the project
You can run the script `ant-colony.py` with [Python 3](https://www.python.org/downloads/): 

```shell
python3 ant-colony.py
```

## Task 1 

### Task 1.1 
Complete the implementation of [`environment.py`](environment.py):
- the constrcutor of the class `Environment` so that the environment is initialized based on the topology of the [att48](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/) TSP instance;
- the function [`initialize_pheromone_map()`](environment.py#L19) for initializing the pheromone trails in the environment based on the Ant System algorithm.
  - HINT: The 3rd-party Python library [tsplib95](https://pypi.org/project/tsplib95/) enables parsing files of the TSPLIB library (e.g. TSP problem and solution files).
  - HINT: The book by (Dorigo and Stützle; 2004) provides directions for the initialization of the pheromone map in the Ant System algorithm \[[1]\].
  
### Task 1.2 
Complete the implementation of [`environment.py`](environment.py):
- the function [`update_pheromone_map()`](environment.py#L23) for updating the pheromone trails in the environment after each cycle based on the Ant System algorithm.

## Task 2

### Task 2.1
Complete the implementation of [`ant.py`](ant.py):
- the function [`get_distance()`](ant.py#L26) for enabling an ant to compute the pseudo-euclidean distance between two cities in the environment;
- the function [`select_path()`](ant.py#L19) for enabling an ant to select the next path to follow based on the Ant System algorithm. 
  - HINT: The specification of the pseudo-euclidean distance algorithm is available in the [TSPLIB95 documentation](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf).
  
### Task 2.2 
Complete the implementation of [`ant.py`](ant.py):
- the function [`run()`](ant.py#L15) for enabling an ant to select and visit locations until it has visited all possible locations of its environment based on the Ant System algorithm.

### Task 2.3 
Complete the implementation of [`ant-colony.py`](ant-colony.py):
- the function [`solve()`](ant-colony.py#L41) for producing a solution for the [att48](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/) TSP instance based on the Ant System algorithm.
  - HINT: The book by (Dorigo and Stützle; 2004) provides directions for the parametrization of the Ant System algorithm \[[1]\].

## Documentation 
- [tsplib95 Python library API documentation](https://tsplib95.readthedocs.io/en/stable/) for working with TSPLIB 95 files (e.g., problem specifications);
- [tsplib95 glossary and documentation](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf) for working with the pseudo-euclidean distance algorithm;
-  For working with the TSP and the Ant System algorithm:
   - \[[1]\] Dorigo, M., & Stützle, T. (2004). Ant colony optimization. The MIT Press.
   - \[[2]\] M. Dorigo, M. Birattari, and T. Stutzle, “Ant colony optimization,” IEEE Computational Intelligence Magazine, vol. 1, no. 4, pp. 28–39, Nov. 2006, doi: 10.1109/mci.2006.329691
- [Lecture slides; slides 22-45](https://learning.unisg.ch/courses/22565/files/3371367?module_item_id=672513).
