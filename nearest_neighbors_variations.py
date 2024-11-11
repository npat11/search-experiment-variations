import numpy as np
import time
import csv
import sys

def read_matrix(file_path):
   with open(file_path, 'r') as file:
       size = int(file.readline().strip())
       matrix_data = [list(map(float, file.readline().strip().split())) for _ in range(size)]
       adjacency_matrix = np.array(matrix_data)
      
   return adjacency_matrix

def nearest_neighbors(adjacency_matrix):
   start_cpu = time.process_time()
   start_real = time.time()
  
   n = adjacency_matrix.shape[0]
   visited = [False] * n
   path = []
   current_node = 0
   path.append(current_node)
   visited[current_node] = True
   nodes_expanded = 0

   for _ in range(n - 1):
       nearest = None
       nearest_distance = float('inf')
      
       for neighbor in range(n):
           if not visited[neighbor] and adjacency_matrix[current_node][neighbor] < nearest_distance:
               nearest_distance = adjacency_matrix[current_node][neighbor]
               nearest = neighbor
      
       path.append(nearest)
       visited[nearest] = True
       current_node = nearest
       nodes_expanded += 1

   path.append(path[0])  # return to starting node
   cost = calculate_cost(adjacency_matrix, path)
  
   end_cpu = time.process_time()
   end_real = time.time()

   return cost, nodes_expanded, end_cpu - start_cpu, end_real - start_real

def calculate_cost(adjacency_matrix, path):
   cost = 0
   for i in range(len(path) - 1):
       cost += adjacency_matrix[path[i]][path[i + 1]]
   return cost

def two_opt(path, adjacency_matrix):
   start_cpu = time.process_time()
   start_real = time.time()
  
   best_cost = calculate_cost(adjacency_matrix, path)
   improved = True
  
   while improved:
       improved = False
       for i in range(1, len(path) - 2):
           for j in range(i + 1, len(path) - 1):
               if j - i == 1: continue
               new_path = path[:]
               new_path[i:j + 1] = reversed(path[i:j + 1])
               new_cost = calculate_cost(adjacency_matrix, new_path)
               if new_cost < best_cost:
                   path = new_path
                   best_cost = new_cost
                   improved = True
                  
   end_cpu = time.process_time()
   end_real = time.time()

   return path, best_cost, end_cpu - start_cpu, end_real - start_real

def nearest_neighbors_with_2opt(adjacency_matrix):
   nn_cost, nodes_expanded, nn_cpu, nn_real = nearest_neighbors(adjacency_matrix)
   optimized_path, optimized_cost, opt_cpu, opt_real = two_opt(list(range(nodes_expanded)), adjacency_matrix)
   return optimized_cost, nodes_expanded, nn_cpu + opt_cpu, nn_real + opt_real

def repeated_randomized_nn_with_2opt(adjacency_matrix, iterations=100):
    best_cost = float('inf')
    n = adjacency_matrix.shape[0]
    total_cpu_time = 0
    total_real_time = 0
    best_nodes_expanded = 0

    for _ in range(iterations):
        nn_cost, nodes_expanded, nn_cpu, nn_real = nearest_neighbors(adjacency_matrix)
        optimized_path, optimized_cost, opt_cpu, opt_real = two_opt(list(range(nodes_expanded)), adjacency_matrix)

        total_cpu_time += nn_cpu + opt_cpu
        total_real_time += nn_real
        
        if optimized_cost < best_cost:
            best_cost = optimized_cost
            best_nodes_expanded = nodes_expanded

    return best_cost, best_nodes_expanded, total_cpu_time, total_real_time

def main():
   function_choice = sys.argv[1]
   adjacency_matrix = read_matrix(sys.argv[2])

   if function_choice == 'nn':
       cost, nodes_expanded, cpu_time, real_time = nearest_neighbors(adjacency_matrix)
   elif function_choice == 'nn2opt':
       cost, nodes_expanded, cpu_time, real_time = nearest_neighbors_with_2opt(adjacency_matrix)
   elif function_choice == 'rnn2opt':
       cost, nodes_expanded, cpu_time, real_time = repeated_randomized_nn_with_2opt(adjacency_matrix)

   if function_choice == 'nn':
       with open('nn.csv', mode='a', newline='') as file:
           writer = csv.writer(file)
           writer.writerow([len(adjacency_matrix), nodes_expanded, cost, cpu_time, real_time])
   elif function_choice == 'nn2opt':
       with open('nn2opt.csv', mode='a', newline='') as file:
           writer = csv.writer(file)
           writer.writerow([len(adjacency_matrix), nodes_expanded, cost, cpu_time, real_time])
   elif function_choice == 'rnn2opt':
       with open('rnn2opt.csv', mode='a', newline='') as file:
           writer = csv.writer(file)
           writer.writerow([len(adjacency_matrix), nodes_expanded, cost, cpu_time, real_time])

if __name__ == "__main__":
   main()

# command: python3 part1.py [choose 1 of nn, nn2opt, rnn2opt] [file relative path]



