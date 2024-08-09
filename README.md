# Bio-Inspired Algorithms for Optimization Problems

This repository contains implementations of several bio-inspired algorithms applied to different optimization problems. The projects included are:

1. **Genetic and Memetic Algorithms**
2. **Min-Max Ant System**
3. **Simulated Annealing**
4. **Particle Swarm Optimization**

## 1. Genetic and Memetic Algorithms

### Projects:
- **Curve Fitting Problem:** 
  - **Objective:** Find the best-fitting curve for a given set of data points.
  - **Approach:** A Genetic Algorithm (GA) is implemented to evolve a population of candidate solutions by applying selection, crossover, and mutation operations. The best individual represents the curve that best fits the data.

- **Travelling Salesman Problem (TSP):**
  - **Objective:** Find the shortest possible route that visits a set of cities and returns to the origin city.
  - **Approach:** Both Genetic Algorithm and Memetic Algorithm (which combines GA with local search) are used. The algorithms evolve potential solutions by simulating natural selection, crossover, and mutation, aiming to minimize the total travel distance.



## 2. Min-Max Ant System

### Project:
- **Set Covering Problem (SCP):**
  - **Objective:** Identify the smallest subset of sets that covers all elements in a universe of elements.
  - **Approach:** The Min-Max Ant System is a variant of the Ant Colony Optimization (ACO) algorithm. Ants traverse the problem space, laying down pheromones to guide other ants toward optimal solutions.
   The algorithm adjusts the pheromone levels to avoid local optima and converge on a global solution.


## 3. Simulated Annealing

### Project:
- **Quadratic Assignment Problem (QAP):**
  - **Objective:** Assign a set of facilities to a set of locations in a way that minimizes the total cost associated with the distances and flow between the facilities.
  - **Approach:** Simulated Annealing (SA) is used to explore the solution space. The algorithm probabilistically accepts worse solutions in the hope of escaping local optima, with the probability of accepting worse solutions decreasing as the "temperature" lowers.


## 4. Particle Swarm Optimization

### Project:
- **Optimization of Mathematical Functions:**
  - **Objective:** Find the minimum and maximum values of various mathematical functions.
  - **Approach:** Particle Swarm Optimization (PSO) simulates a swarm of particles that search the solution space by adjusting their positions and velocities based on personal and group experience. The swarm converges toward optimal solutions over time.

