from flask import Flask, request, jsonify
import math
from collections import defaultdict

from routes import app

class TradingArbitrageFinder:
    def __init__(self, goods, ratios):
        self.goods = goods
        self.num_goods = len(goods)
        self.graph = defaultdict(list)
        self.build_graph(ratios)
    
    def build_graph(self, ratios):
        """Build a graph where edges represent trading ratios"""
        for ratio in ratios:
            from_good, to_good, rate = int(ratio[0]), int(ratio[1]), float(ratio[2])
            # Store the trading rate from from_good to to_good
            self.graph[from_good].append((to_good, rate))
    
    def find_arbitrage_bellman_ford(self):
        """
        Use Bellman-Ford algorithm to detect positive cycles (arbitrage opportunities)
        We negate the logarithms to find positive cycles instead of negative cycles
        """
        # Use negative log to convert multiplication to addition and find positive cycles
        INF = float('inf')
        
        best_cycle = None
        max_gain = 0
        
        # Try starting from each good
        for start_good in range(self.num_goods):
            # Initialize distances - we want to maximize gain so use negative log
            dist = [INF] * self.num_goods
            predecessor = [-1] * self.num_goods
            dist[start_good] = 0
            
            # Relax edges for num_goods - 1 iterations
            for _ in range(self.num_goods - 1):
                for from_good in self.graph:
                    if dist[from_good] == INF:
                        continue
                    for to_good, rate in self.graph[from_good]:
                        # Use negative log to find positive cycles
                        new_dist = dist[from_good] - math.log(rate)
                        if new_dist < dist[to_good]:
                            dist[to_good] = new_dist
                            predecessor[to_good] = from_good
            
            # Check for positive cycles (negative cycles in our transformed graph)
            for from_good in self.graph:
                if dist[from_good] == INF:
                    continue
                for to_good, rate in self.graph[from_good]:
                    new_dist = dist[from_good] - math.log(rate)
                    if new_dist < dist[to_good]:
                        # Found a positive cycle, extract it
                        cycle = self.extract_cycle(predecessor, to_good)
                        if cycle:
                            gain = self.calculate_cycle_gain(cycle)
                            if gain > max_gain:
                                max_gain = gain
                                best_cycle = cycle
        
        return best_cycle, max_gain
    
    def extract_cycle(self, predecessor, start_node):
        """Extract a cycle from the predecessor array"""
        visited = set()
        current = start_node
        
        # Find a node that's definitely in the cycle
        for _ in range(self.num_goods):
            current = predecessor[current]
            if current == -1:
                return None
        
        # Extract the cycle
        cycle_start = current
        cycle = [current]
        current = predecessor[current]
        
        while current != cycle_start:
            cycle.append(current)
            current = predecessor[current]
            if len(cycle) > self.num_goods:  # Safety check
                break
        
        cycle.reverse()
        return cycle
    
    def calculate_cycle_gain(self, cycle):
        """Calculate the total gain from a trading cycle"""
        if not cycle or len(cycle) < 2:
            return 0
        
        total_rate = 1.0
        
        # Calculate the total multiplication factor for the cycle
        for i in range(len(cycle)):
            from_good = cycle[i]
            to_good = cycle[(i + 1) % len(cycle)]
            
            # Find the rate from from_good to to_good
            rate_found = False
            for neighbor, rate in self.graph[from_good]:
                if neighbor == to_good:
                    total_rate *= rate
                    rate_found = True
                    break
            
            if not rate_found:
                return 0  # Invalid cycle
        
        # Return gain as percentage (total_rate - 1) * 100
        return (total_rate - 1) * 100
    
    def find_all_cycles_dfs(self, max_depth=4):
        """Find all cycles using DFS approach (alternative method)"""
        best_cycle = None
        max_gain = 0
        
        def dfs(current_good, path, visited_in_path, total_rate, depth):
            nonlocal best_cycle, max_gain
            
            if depth > max_depth:
                return
            
            if current_good in visited_in_path and len(path) > 2:
                # Found a cycle
                cycle_start_idx = path.index(current_good)
                cycle = path[cycle_start_idx:]
                gain = (total_rate - 1) * 100
                
                if gain > max_gain:
                    max_gain = gain
                    best_cycle = cycle[:]
                return
            
            if current_good in visited_in_path:
                return
            
            visited_in_path.add(current_good)
            
            for next_good, rate in self.graph[current_good]:
                if next_good not in visited_in_path or (next_good in path and len(path) > 2):
                    new_path = path + [next_good]
                    new_total_rate = total_rate * rate
                    dfs(next_good, new_path, visited_in_path.copy(), new_total_rate, depth + 1)
        
        # Try starting from each good
        for start_good in range(self.num_goods):
            dfs(start_good, [start_good], set(), 1.0, 0)
        
        return best_cycle, max_gain
    
    def find_best_arbitrage(self):
        """Find the best arbitrage opportunity using multiple methods"""
        # Try Bellman-Ford first
        cycle1, gain1 = self.find_arbitrage_bellman_ford()
        
        # Try DFS as backup
        cycle2, gain2 = self.find_all_cycles_dfs()
        
        # Return the better result
        if gain1 >= gain2:
            return cycle1, gain1
        else:
            return cycle2, gain2
    
    def format_result(self, cycle, gain):
        """Format the result with good names"""
        if not cycle:
            return None
        
        # Convert indices to good names
        path = [self.goods[i] for i in cycle]
        # Add the starting good at the end to show the complete cycle
        if len(path) > 0:
            path.append(path[0])
        
        return {
            "path": path,
            "gain": gain
        }

@app.route('/The-Ink-Archive', methods=['POST'])
def analyze_data():
    try:
        trading_data = request.json
        
        results = []
        
        # Analyze the first trading scenario
        scenario = trading_data[0]
        finder = TradingArbitrageFinder(scenario['goods'], scenario['ratios'])
        cycle, gain = finder.find_best_arbitrage()
        
        if cycle and gain > 0:
            result = finder.format_result(cycle, gain)
            results.append(result)
        else:
            results.append({"message": "No arbitrage opportunities found", "path": [], "gain": 0})
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)