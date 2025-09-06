from flask import request, jsonify
import heapq
from collections import defaultdict
import os
import logging

from routes import app

logger = logging.getLogger(__name__)

def solve_princess_diaries(tasks, subway, start_station):
    """Optimized and correct solution"""
    if not tasks:
        return 0, 0, []
    
    # Build graph efficiently
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
    
    # Convert to list for indexing
    station_list = sorted(stations)
    station_to_idx = {s: i for i, s in enumerate(station_list)}
    n_stations = len(station_list)
    
    # Floyd-Warshall for small graphs, Dijkstra for larger ones
    if n_stations <= 50:
        # Use Floyd-Warshall for small graphs
        INF = float('inf')
        dist = [[INF] * n_stations for _ in range(n_stations)]
        
        # Initialize
        for i in range(n_stations):
            dist[i][i] = 0
        
        # Set edges
        for station in graph:
            i = station_to_idx[station]
            for neighbor, cost in graph[station]:
                j = station_to_idx[neighbor]
                dist[i][j] = min(dist[i][j], cost)
        
        # Floyd-Warshall
        for k in range(n_stations):
            for i in range(n_stations):
                for j in range(n_stations):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        def get_distance(u, v):
            return dist[station_to_idx[u]][station_to_idx[v]]
    else:
        # Use memoized Dijkstra for larger graphs
        distance_cache = {}
        
        def dijkstra(start):
            if start in distance_cache:
                return distance_cache[start]
            
            dist = {s: float('inf') for s in stations}
            dist[start] = 0
            pq = [(0, start)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                
                for v, w in graph[u]:
                    if d + w < dist[v]:
                        dist[v] = d + w
                        heapq.heappush(pq, (d + w, v))
            
            distance_cache[start] = dist
            return dist
        
        def get_distance(u, v):
            return dijkstra(u)[v]
    
    # Sort tasks by end time
    tasks.sort(key=lambda x: x['end'])
    n = len(tasks)
    
    # Precompute latest compatible task for each task
    compatible = [-1] * n
    for i in range(n):
        # Binary search for latest compatible task
        left, right = 0, i - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if tasks[mid]['end'] <= tasks[i]['start']:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        compatible[i] = result
    
    # DP: for each task, store (max_score, min_cost, task_indices)
    dp = [None] * n
    
    # Base case: first task
    cost0 = get_distance(start_station, tasks[0]['station']) + get_distance(tasks[0]['station'], start_station)
    dp[0] = (tasks[0]['score'], cost0, [0])
    
    # Also consider not taking first task
    best_so_far = (0, 0, [])
    if dp[0][0] > 0:
        best_so_far = dp[0]
    
    for i in range(1, n):
        current_task = tasks[i]
        
        # Option 1: Don't take current task (inherit best so far)
        dp[i] = best_so_far
        
        # Option 2: Take current task
        prev_idx = compatible[i]
        
        if prev_idx == -1:
            # No compatible previous tasks
            cost = get_distance(start_station, current_task['station']) + get_distance(current_task['station'], start_station)
            score = current_task['score']
            schedule = [i]
        else:
            # Extend from best compatible solution
            if prev_idx == 0:
                prev_solution = dp[0]
            else:
                # Find best solution up to prev_idx
                prev_solution = (0, 0, [])
                for j in range(prev_idx + 1):
                    if dp[j] and (dp[j][0] > prev_solution[0] or 
                                 (dp[j][0] == prev_solution[0] and dp[j][1] < prev_solution[1])):
                        prev_solution = dp[j]
            
            prev_score, prev_cost, prev_schedule = prev_solution
            
            if prev_schedule:
                # Get last task in previous schedule
                last_task_idx = prev_schedule[-1]
                last_station = tasks[last_task_idx]['station']
                
                # Calculate cost: remove return to start, add travel to current, add new return
                old_return = get_distance(last_station, start_station)
                travel = get_distance(last_station, current_task['station'])
                new_return = get_distance(current_task['station'], start_station)
                
                cost = prev_cost - old_return + travel + new_return
                schedule = prev_schedule + [i]
            else:
                cost = get_distance(start_station, current_task['station']) + get_distance(current_task['station'], start_station)
                schedule = [i]
            
            score = prev_score + current_task['score']
        
        # Update if this option is better
        current_score, current_cost, current_schedule = dp[i]
        if (score > current_score or (score == current_score and cost < current_cost)):
            dp[i] = (score, cost, schedule)
        
        # Update best so far
        if dp[i][0] > best_so_far[0] or (dp[i][0] == best_so_far[0] and dp[i][1] < best_so_far[1]):
            best_so_far = dp[i]
    
    # Find the overall best solution
    best_solution = (0, 0, [])
    for solution in dp:
        if solution and (solution[0] > best_solution[0] or 
                        (solution[0] == best_solution[0] and solution[1] < best_solution[1])):
            best_solution = solution
    
    max_score, min_cost, task_indices = best_solution
    
    # Convert task indices back to names and sort by start time
    if task_indices:
        schedule_with_times = [(tasks[i]['name'], tasks[i]['start']) for i in task_indices]
        schedule_with_times.sort(key=lambda x: x[1])
        schedule = [name for name, _ in schedule_with_times]
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