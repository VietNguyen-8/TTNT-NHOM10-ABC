import random
import datetime
import pandas as pd
import numpy as np
import networkx as nx
# random: Được dùng để thực hiện các thao tác ngẫu nhiên như:
# Chọn ngẫu nhiên hai nút trong danh sách.
# Sinh số ngẫu nhiên cho việc lựa chọn xác suất.
# datetime: Được dùng để tính toán thời gian thực thi của thuật toán.
# pandas: Hỗ trợ lưu trữ kết quả vào dạng bảng (DataFrame) và xuất dữ liệu ra file CSV.
# numpy: Cung cấp công cụ xử lý mảng số học hiệu quả, ví dụ:
# Tạo mảng np.zeros.
# Các phép toán như np.power, np.sum.
# networkx: Được dùng để xây dựng và thao tác với đồ thị (graph), bao gồm:
# Tạo đồ thị có hướng (directed graph).
# Thêm cạnh và trọng số.
# Tìm tất cả các đường đi đơn giản giữa hai nút.


class ArtificialBeeColony:
    def __init__(self, G, num_bees, max_iterations):
        self.G = G
        self.num_bees = num_bees
        self.max_iterations = max_iterations
        
        self.current_population = [self.generate_possible_solution() for i in range(self.num_bees)]
        self.current_best_solution = self.current_population[0]  # initialize best solution as the first element of the population
        self.population_size = len(self.current_population)
        self.num_employeed_bees = self.population_size // 2
        self.num_onlooker_bees = self.population_size - self.num_employeed_bees
        self.test_data = []
        self.test_cases = 0

#    self.G: Đồ thị đầu vào (graph) mà thuật toán sẽ làm việc.
# self.num_bees: Tổng số lượng ong (bao gồm ong thợ và ong do thám).
# self.max_iterations: Số lần lặp tối đa để thuật toán dừng lại.
# self.current_population: Tập hợp các giải pháp ban đầu, mỗi giải pháp là một đường đi ngẫu nhiên trong đồ thị.
# self.current_best_solution: Lưu giải pháp tốt nhất hiện tại (khởi đầu là giải pháp đầu tiên trong danh sách).
# self.num_employeed_bees: Số lượng ong thợ (chiếm một nửa đàn ong).
# self.num_onlooker_bees: Số lượng ong do thám (phần còn lại).
# self.test_data: Lưu dữ liệu về fitness qua từng lần lặp (để phân tích sau).
# self.test_cases: Biến đếm số lần kiểm tra fitness của các giải pháp.

    
    """
    Hàm tình độ phù hợp(fitness)
    """ 
    def evaluate_fitness(self, path, eps=0.9):
        fitness = 0.0
        
        for i in range(1, len(path)):
            total_distance = 0
            curr_node = path[i-1]
            next_node = path[i]
            if self.G.has_edge(curr_node, next_node):
                fitness += self.G[curr_node][next_node]['weight']
            else:
                fitness += 0
        fitness = np.power(abs(fitness + eps), 2)
        return fitness
        
# path: Đường đi cần tính toán độ phù hợp.
# eps: Giá trị nhỏ để tránh chia cho 0 khi tính xác suất.
# Mô tả:
# Lặp qua các cặp nút liên tiếp trong đường đi.
# Cộng trọng số của cạnh giữa mỗi cặp nút vào tổng fitness nếu cạnh tồn tại.
# Tăng độ phù hợp bằng cách sử dụng lũy thừa np.power.

    
    def apply_random_neighborhood_structure(self, path):
        """
        This function applies the neighborhood structure to find a new solution.
        It randomly swaps two nodes in the path.
        """
        new_path = path.copy()
        node1, node2 = random.sample(path, 2)
        node1_index = path.index(node1)
        node2_index = path.index(node2)
        new_path[node1_index], new_path[node2_index] = new_path[node2_index], new_path[node1_index]
        
        return new_path
    """
    Hàm tạo giải pháp hàng xóm
    """ 
    def sort_population_by_fitness(self, population):
        """
        This function sorts the population of paths based on their fitness (the total weight of the edges in the path)
        """
        return sorted(population, key=lambda x: self.evaluate_fitness(x), reverse=True)
        
# Mô tả:
# Sao chép đường đi hiện tại để giữ nguyên bản gốc.
# Chọn ngẫu nhiên hai nút từ đường đi.
# Hoán đổi vị trí hai nút trong đường đi để tạo ra một giải pháp mới.

        """
        Hàm chọn giải pháp theo xác suất
        """
    def choose_solution_with_probability(self, population, probability_list):
        
        random_value = random.random()
        cumulative_probability = 0.0
        for i in range(len(population)):
            cumulative_probability += probability_list[i]
            if random_value <= cumulative_probability:
                return population[i]
                
#  Mục đích: Chọn một giải pháp trong dân số dựa trên xác suất.
# Mô tả:
# Tính cumulative probability (xác suất tích lũy).
# So sánh với một giá trị ngẫu nhiên random_value.
# Trả về giải pháp tương ứng khi giá trị ngẫu nhiên nằm trong khoảng xác suất.


        """
        Hàm sinh giải pháp ngẫu nhiên
        """
    def generate_possible_solution(self):
        nodes = list(self.G.nodes)
        start = nodes[0]
        end = nodes[-1]
        samples = list(nx.all_simple_paths(self.G, start, end))
        for i in range(len(samples)):
            if len(samples[i]) != len(nodes):
                extra_nodes = [node for node in nodes if node not in samples[i]]
                random.shuffle(extra_nodes)
                samples[i] = samples[i] + extra_nodes

        sample_node = random.choice(samples)
        return sample_node
# Mục đích: Tạo ra một đường đi ngẫu nhiên từ nút bắt đầu đến nút kết thúc.
# Mô tả:
# Lấy tất cả các đường đi đơn giản từ nút đầu đến nút cuối.
# Nếu một đường đi không chứa đủ các nút, bổ sung thêm các nút còn thiếu vào cuối đường đi.
# Trả về một đường đi ngẫu nhiên.

        """
        Thuật toán chính
        """
    def run(self, patience=10):
        gen_fitness = np.zeros(self.max_iterations)
        patience_counter = 0
        for iteration in range(self.max_iterations):
            # Employed Bee phase
            for i in range(self.num_employeed_bees):
                current_solution = self.current_population[i]
                new_solution = self.apply_random_neighborhood_structure(current_solution)
                new_solution_cost = self.evaluate_fitness(new_solution)
                current_solution_cost = self.evaluate_fitness(current_solution)

                if new_solution_cost > current_solution_cost:
                    self.current_population[i] = new_solution
                    self.test_cases += 1

            #Onlooker Bee phase
            probability_list = [1.0 / self.evaluate_fitness(solution) for solution in self.current_population]
            probability_list = [probability / sum(probability_list) for probability in probability_list]

            for i in range(self.num_onlooker_bees):
                selected_solution = self.choose_solution_with_probability(self.current_population, probability_list)
                new_solution = self.apply_random_neighborhood_structure(selected_solution)
                new_solution_cost = self.evaluate_fitness(new_solution)
                selected_solution_cost = self.evaluate_fitness(selected_solution)

                if new_solution_cost > selected_solution_cost:
                    selected_solution_index = self.current_population.index(selected_solution)
                    self.current_population[selected_solution_index] = new_solution
                    self.test_cases += 1

            # Scout Bee phase
            current_population = self.sort_population_by_fitness(self.current_population)
            current_fitness_value = self.evaluate_fitness(self.current_best_solution)
            
            if self.evaluate_fitness(self.current_population[0]) > current_fitness_value:
                self.current_best_solution = self.current_population[0]

            # If the best solution does not change for a certain number of iterations, generate a new random solution
            gen_fitness[iteration] = current_fitness_value
            
            if iteration > 0:
                if gen_fitness[iteration]==gen_fitness[iteration-1]:
                    patience_counter += 1
                    
            if patience_counter >= patience:
                self.current_population[-1] = self.generate_possible_solution()
                patience_counter = 0
            
            self.test_data.append([iteration, current_fitness_value, self.test_cases])
        
            last_node = list(self.G.nodes)[-1]
        last_node_index = self.current_best_solution.index(last_node) + 1
        return self.current_best_solution[ : last_node_index], current_fitness_value
    
if __name__ == "__main__":
    """ Example Setup """
    Gn = nx.DiGraph()
    
    #Add nodes to the graph
    for i in range(11):
        Gn.add_node(i)

    #Add Weighted nodes to the graph
    edges = [(0, 1,{'weight': 1}), (1, 3,{'weight': 2}), (1, 2,{'weight': 1}),(2, 4,{'weight': 2}),
            (3, 2,{'weight': 2}),(3, 4,{'weight': 1}),(3, 5,{'weight': 2}),(3, 7,{'weight': 4}),
            (4, 5,{'weight': 1}),(4, 6,{'weight': 2}),(5, 7,{'weight': 2}),(5, 8,{'weight': 3}),
            (6, 7,{'weight': 1}),(7, 9,{'weight': 2}),(8, 10,{'weight': 2}),(9, 10,{'weight': 1})]

    Gn.add_edges_from(edges)
    abc = ArtificialBeeColony(Gn, num_bees = 53, max_iterations=500)

    start = datetime.datetime.now()
    best_path, best_fitness = abc.run(patience = 12)
    end = datetime.datetime.now()

    abc_time = end - start

    abc_test_data = pd.DataFrame(abc.test_data, columns = ["iterations","fitness_value","test_cases"])

    print("Optimal path: ", best_path)
    print("Optimal path cost: ", best_fitness)
    print("ABC total Exec time => ", abc_time.total_seconds())
    abc_test_data.to_csv("abc_test_data_results.csv")
