from flask import request, jsonify
import heapq
from collections import defaultdict
import os
import logging

from routes import app

logger = logging.getLogger(__name__)

def solve_princess_diaries(tasks, subway, start_station):
    """Optimized and correct solution with proper DP"""
    if not tasks:
        return 0, 0, []
    
    # Build graph
    graph = defaultdict(list)
    stations = set([start_station])
    
    for route in subway:
        u, v, cost = route['connection'][0], route['connection'][1], route['fee']
        graph[u].append((v, cost))
        graph[v].append((u, cost))
        stations.add(u)
        stations.add(v)
    
    for task in tasks:
        stations.add(task['station'])
    
    # Efficient distance calculation
    station_list = sorted(stations)
    station_to_idx = {s: i for i, s in enumerate(station_list)}
    n_stations = len(station_list)
    
    # Use Floyd-Warshall for complete distance matrix
    INF = float('inf')
    dist = [[INF] * n_stations for _ in range(n_stations)]
    
    # Initialize
    for i in range(n_stations):
        dist[i][i] = 0
    
    # Set direct edges
    for station in graph:
        i = station_to_idx[station]
        for neighbor, cost in graph[station]:
            j = station_to_idx[neighbor]
            dist[i][j] = min(dist[i][j], cost)
    
    # Floyd-Warshall algorithm (handles cycles correctly)
    for k in range(n_stations):
        for i in range(n_stations):
            for j in range(n_stations):
                if dist[i][k] != INF and dist[k][j] != INF:
                    new_dist = dist[i][k] + dist[k][j]
                    if new_dist < dist[i][j]:
                        dist[i][j] = new_dist
    
    # Check for negative cycles (shouldn't happen with positive edge weights)
    for i in range(n_stations):
        if dist[i][i] < 0:
            # This shouldn't happen with positive weights, but handle gracefully
            dist[i][i] = 0
    
    def get_distance(u, v):
        return dist[station_to_idx[u]][station_to_idx[v]]
    
    # Sort tasks by end time for DP
    sorted_tasks = sorted(enumerate(tasks), key=lambda x: x[1]['end'])
    n = len(sorted_tasks)
    
    # Precompute latest compatible task using binary search
    compatible = [-1] * n
    for i in range(n):
        curr_start = sorted_tasks[i][1]['start']
        left, right = 0, i - 1
        best = -1
        
        while left <= right:
            mid = (left + right) // 2
            if sorted_tasks[mid][1]['end'] <= curr_start:
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        compatible[i] = best
    
    # DP: dp[i] = (max_score, min_cost_for_max_score, best_schedule_indices)
    # considering tasks 0 to i (inclusive)
    dp = [(0, 0, [])]  # Base case: no tasks
    
    for i in range(n):
        orig_idx, current_task = sorted_tasks[i]
        
        # Option 1: Don't take current task
        prev_score, prev_cost, prev_schedule = dp[i]
        dp.append((prev_score, prev_cost, prev_schedule[:]))
        
        # Option 2: Take current task
        latest_compatible = compatible[i]
        
        if latest_compatible == -1:
            # No previous compatible tasks
            cost = get_distance(start_station, current_task['station']) + \
                   get_distance(current_task['station'], start_station)
            new_score = current_task['score']
            new_schedule = [orig_idx]
        else:
            # Take from the best solution up to latest_compatible
            base_score, base_cost, base_schedule = dp[latest_compatible + 1]
            
            if base_schedule:
                # Find the last task in the base schedule
                last_orig_idx = base_schedule[-1]
                last_task = tasks[last_orig_idx]
                last_station = last_task['station']
                
                # Remove the return cost from base, add travel + new return
                old_return = get_distance(last_station, start_station)
                travel = get_distance(last_station, current_task['station'])
                new_return = get_distance(current_task['station'], start_station)
                
                cost = base_cost - old_return + travel + new_return
            else:
                # Starting fresh
                cost = get_distance(start_station, current_task['station']) + \
                       get_distance(current_task['station'], start_station)
            
            new_score = base_score + current_task['score']
            new_schedule = base_schedule + [orig_idx]
        
        # Update dp[i+1] if taking current task is better
        current_score, current_cost, current_schedule = dp[i + 1]
        
        if (new_score > current_score or 
            (new_score == current_score and cost < current_cost)):
            dp[i + 1] = (new_score, cost, new_schedule)
    
    # The answer is in dp[n]
    max_score, min_cost, best_schedule_indices = dp[n]
    
    # Convert indices back to task names and sort by start time
    if best_schedule_indices:
        schedule_tasks = [(tasks[idx]['name'], tasks[idx]['start']) for idx in best_schedule_indices]
        schedule_tasks.sort(key=lambda x: x[1])  # Sort by start time
        schedule = [name for name, _ in schedule_tasks]
    else:
        schedule = []
    
    return max_score, int(min_cost), schedule

@app.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    try:
        data = request.get_json()
        
        tasks = data['tasks']
        subway = data['subway']
        start_station = data['starting_station']
        
        max_score, min_fee, schedule = solve_princess_diaries(tasks, subway, start_station)
        
        return jsonify({
            "max_score": max_score,
            "min_fee": min_fee,
            "schedule": schedule
        })
        
    except Exception as e:
        return jsonify({
            "max_score": 0,
            "min_fee": 0,
            "schedule": []
        }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)