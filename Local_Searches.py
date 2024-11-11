#don't need nodes expanded
import numpy as np
import time
import csv
import sys

def create_population(size, num_cities):
    return [np.random.permutation(num_cities) for _ in range(size)]

def calculate_cost(tour, adj_matrix):
    cost = sum(adj_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    cost += adj_matrix[tour[-1], tour[0]]  # Return to starting city
    return cost

def select_parents(population, num_parents, adj_matrix):
    costs = [calculate_cost(tour, adj_matrix) for tour in population]
    selected_indices = np.argsort(costs)[:num_parents]
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2, crossover_length):
    size = len(parent1)
    start = np.random.randint(0, size - crossover_length + 1)
    child = [-1] * size
    child[start:start + crossover_length] = parent1[start:start + crossover_length]

    current_pos = (start + crossover_length) % size
    for city in parent2:
        if city not in child:
            child[current_pos] = city
            current_pos = (current_pos + 1) % size

    return child

def mutate(tour, mutation_rate):
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(tour), 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]

def genetic_algorithm(adj_matrix):
    population_size = 200
    generations = 500
    mutation_rate = 0.01
    crossover_probability = 0.9
    crossover_length = 3
    num_parents = 20

    num_cities = len(adj_matrix)
    population = create_population(population_size, num_cities)

    for _ in range(generations):
        population.sort(key=lambda tour: calculate_cost(tour, adj_matrix))
        next_generation = select_parents(population, num_parents, adj_matrix)

        while len(next_generation) < population_size:
            parents = np.array(next_generation)
            if np.random.rand() < crossover_probability:
                parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
                child = crossover(parent1, parent2, crossover_length)
            else:  # Clone 1 parent
                child = parents[np.random.choice(len(parents))].copy()

            mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation

    best_tour = min(population, key=lambda tour: calculate_cost(tour, adj_matrix))
    best_cost = calculate_cost(best_tour, adj_matrix)

    return best_tour, best_cost

def hill_climbing(adj_matrix, max_restarts=10):
    num_cities = len(adj_matrix)
    best_cost = float('inf')
    best_tour = None

    for _ in range(max_restarts):
        current_tour = np.random.permutation(num_cities)
        current_cost = calculate_cost(current_tour, adj_matrix)

        while True:
            improved = False
            for i in range(num_cities):
                for j in range(i + 1, num_cities):
                    new_tour = current_tour.copy()
                    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                    new_cost = calculate_cost(new_tour, adj_matrix)
                    if new_cost < current_cost:
                        current_tour, current_cost = new_tour, new_cost
                        improved = True
            if not improved:
                break
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour

    return best_tour, best_cost

def simulated_annealing(adj_matrix, initial_temp=1000, cooling_rate=0.995, max_restarts=10):
    num_cities = len(adj_matrix)
    best_cost = float('inf')
    best_tour = None

    for _ in range(max_restarts):
        current_tour = np.random.permutation(num_cities)
        current_cost = calculate_cost(current_tour, adj_matrix)
        temp = initial_temp

        while temp > 1:
            new_tour = current_tour.copy()
            i, j = np.random.choice(num_cities, 2, replace=False)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_cost = calculate_cost(new_tour, adj_matrix)

            if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / temp):
                current_tour, current_cost = new_tour, new_cost
            
            temp *= cooling_rate

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour

    return best_tour, best_cost

def read_matrix(file_path):
    with open(file_path, 'r') as file:
        size = int(file.readline().strip())
        matrix_data = [list(map(float, file.readline().strip().split())) for _ in range(size)]
    return np.array(matrix_data)

def main():
    algorithm = sys.argv[1]
    adjacency_matrix = read_matrix(sys.argv[2])

    start_cpu = time.process_time()
    start_real = time.time()

    if algorithm == "ga":
        best_tour, best_cost = genetic_algorithm(adjacency_matrix)
    elif algorithm == "hc":
        best_tour, best_cost = hill_climbing(adjacency_matrix)
    elif algorithm == "sa":
        best_tour, best_cost = simulated_annealing(adjacency_matrix)

    end_cpu = time.process_time()
    end_real = time.time()
    cpu_time = end_cpu - start_cpu
    real_time = end_real - start_real

    with open(f'{algorithm}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([len(adjacency_matrix), best_cost, cpu_time, real_time])

if __name__ == "__main__":
    main()

# command: python3 part3.py [choose ga, hc, sa] [relative file path]
