#corrected

import numpy as np
import csv
import sys
import time
import heapq

def read_matrix(file_path):
    with open(file_path, 'r') as file:
        size = int(file.readline().strip())
        matrix_data = [list(map(float, file.readline().strip().split())) for _ in range(size)]
        adjacency_matrix = np.array(matrix_data)
    return adjacency_matrix

def prim_mst(adjacency_matrix):
    size = len(adjacency_matrix)
    visited = [False] * size
    min_edge = [(0, 0)]  # (weight, vertex)
    total_weight = 0
    num_edges = 0

    while min_edge and num_edges < size:
        weight, u = heapq.heappop(min_edge)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += weight
        num_edges += 1

        for v in range(size):
            if not visited[v] and adjacency_matrix[u][v] > 0:
                heapq.heappush(min_edge, (adjacency_matrix[u][v], v))

    return total_weight

class AStarNode:
    def __init__(self, city, path, cost, heuristic):
        self.city = city
        self.path = path
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def a_star_tsp(adjacency_matrix):
    size = len(adjacency_matrix)
    start_city = 0
    initial_path = [start_city]
    initial_cost = 0
    heuristic = prim_mst(adjacency_matrix)
    start_node = AStarNode(start_city, initial_path, initial_cost, heuristic)

    priority_queue = []
    heapq.heappush(priority_queue, start_node)
    nodes_expanded = 0
    optimal_cost = float('inf')
    
    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        nodes_expanded += 1

        if len(current_node.path) == size:
            final_cost = current_node.cost + adjacency_matrix[current_node.city][start_city]
            optimal_cost = min(optimal_cost, final_cost)
            continue

        for next_city in range(size):
            if next_city not in current_node.path:
                new_path = current_node.path + [next_city]
                new_cost = current_node.cost + adjacency_matrix[current_node.city][next_city]
                heuristic = prim_mst(adjacency_matrix) - sum(adjacency_matrix[new_path[i]][new_path[i + 1]] for i in range(len(new_path) - 1))
                new_node = AStarNode(next_city, new_path, new_cost, heuristic)
                heapq.heappush(priority_queue, new_node)

    return optimal_cost, nodes_expanded

def main():
    file_path = sys.argv[1]
    adjacency_matrix = read_matrix(file_path)

    start_time_cpu = time.process_time()
    start_time_real = time.time()

    optimal_cost, nodes_expanded = a_star_tsp(adjacency_matrix)

    end_time_cpu = time.process_time()
    end_time_real = time.time()

    with open('a_star.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([len(adjacency_matrix), nodes_expanded, optimal_cost, end_time_cpu - start_time_cpu, end_time_real - start_time_real])

if __name__ == "__main__":
    main()

#command: python3 part2.py [file relative path]