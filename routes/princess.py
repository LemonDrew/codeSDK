from flask import Flask, request, jsonify
import heapq
from collections import defaultdict
from functools import lru_cache

from routes import app

def floyd_warshall(graph, stations):
    """Precompute all-pairs shortest paths using Floyd-Warshall"""
    # Initialize distance matrix
    dist = {}
    for i in stations:
        for j in stations:
            if i == j:
                dist[(i, j)] = 0
            else:
                dist[(i, j)] = float('inf')
    
    # Set direct edge weights
    for station in graph:
        for neighbor, weight in graph[station]:
            dist[(station, neighbor)] = weight
    
    # Floyd-Warshall algorithm
    for k in stations:
        for i in stations:
            for j in stations:
                if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
    
    return dist

def build_graph(subway_routes):
    """Build adjacency list representation of the subway system"""
    graph = defaultdict(list)
    stations = set()
    
    for route in subway_routes:
        station1, station2 = route['connection']
        fee = route['fee']
        graph[station1].append((station2, fee))
        graph[station2].append((station1, fee))
        stations.add(station1)
        stations.add(station2)
    
    return graph, stations

def find_optimal_schedule(tasks, subway_routes, starting_station):
    """Find the schedule that maximizes score and minimizes transport cost"""
    
    if not tasks:
        return 0, 0, []
    
    # Build subway graph and get all stations
    graph, graph_stations = build_graph(subway_routes)
    
    # Add starting station and task stations
    all_stations = graph_stations.copy()
    all_stations.add(starting_station)
    for task in tasks:
        all_stations.add(task['station'])
    
    # Precompute all shortest distances using Floyd-Warshall
    distances = floyd_warshall(graph, all_stations)
    
    # Sort tasks by end time for easier processing
    sorted_tasks = sorted(tasks, key=lambda x: x['end'])
    n = len(sorted_tasks)
    
    # DP: dp[i] = (max_score, min_cost_for_max_score, best_schedule)
    # for the best schedule considering tasks 0 to i
    dp = [(0, 0, [])]  # Base case: no tasks selected
    
    for i in range(n):
        current_task = sorted_tasks[i]
        
        # Option 1: Don't take current task
        prev_score, prev_cost, prev_schedule = dp[i]
        dp.append((prev_score, prev_cost, prev_schedule))
        
        # Option 2: Take current task
        # Find the latest non-overlapping task
        best_prev_idx = -1
        for j in range(i - 1, -1, -1):
            if sorted_tasks[j]['end'] <= current_task['start']:
                best_prev_idx = j
                break
        
        if best_prev_idx == -1:
            # No previous tasks, start from beginning
            cost = distances[(starting_station, current_task['station'])] + \
                   distances[(current_task['station'], starting_station)]
            new_score = current_task['score']
            new_schedule = [current_task['name']]
        else:
            # Extend from best previous schedule
            prev_score, prev_cost, prev_schedule = dp[best_prev_idx + 1]
            
            if prev_schedule:
                # Find the last task in previous schedule
                last_task_name = prev_schedule[-1]
                last_task = next(t for t in tasks if t['name'] == last_task_name)
                last_station = last_task['station']
                
                # Calculate cost: remove return to start, add travel to current, add return
                cost = prev_cost - distances[(last_station, starting_station)] + \
                       distances[(last_station, current_task['station'])] + \
                       distances[(current_task['station'], starting_station)]
            else:
                cost = distances[(starting_station, current_task['station'])] + \
                       distances[(current_task['station'], starting_station)]
            
            new_score = prev_score + current_task['score']
            new_schedule = prev_schedule + [current_task['name']]
        
        # Update dp[i+1] if this option is better
        current_best_score, current_best_cost, current_best_schedule = dp[i + 1]
        
        if (new_score > current_best_score or 
            (new_score == current_best_score and cost < current_best_cost)):
            dp[i + 1] = (new_score, cost, new_schedule)
    
    max_score, min_cost, best_schedule = dp[n]
    
    # Sort schedule by start time
    if best_schedule:
        task_start_times = {task['name']: task['start'] for task in tasks}
        best_schedule.sort(key=lambda name: task_start_times[name])
    
    return max_score, min_cost, best_schedule

@app.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    try:
        data = request.get_json()
        
        tasks = data['tasks']
        subway_routes = data['subway']
        starting_station = data['starting_station']
        
        max_score, min_fee, schedule = find_optimal_schedule(tasks, subway_routes, starting_station)
        
        return jsonify({
            "max_score": max_score,
            "min_fee": min_fee,
            "schedule": schedule
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
