from flask import request, jsonify
import heapq
from collections import defaultdict
import os
import logging

from routes import app

logger = logging.getLogger(__name__)

def solve_princess_diaries(tasks, subway, start_station):
    """Ultra-optimized Python solution"""
    if not tasks:
        return 0, 0, []
    
    # Build adjacency list
    graph = defaultdict(list)
    all_stations = {start_station}
    
    for route in subway:
        u, v, cost = route['connection'][0], route['connection'][1], route['fee']
        graph[u].append((v, cost))
        graph[v].append((u, cost))
        all_stations.add(u)
        all_stations.add(v)
    
    for task in tasks:
        all_stations.add(task['station'])
    
    # Optimized single-source shortest path
    def dijkstra(start):
        dist = {node: float('inf') for node in all_stations}
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
        
        return dist
    
    # Precompute distances only for stations we need
    needed_stations = {start_station}
    for task in tasks:
        needed_stations.add(task['station'])
    
    distances = {}
    for station in needed_stations:
        dist_map = dijkstra(station)
        for target in needed_stations:
            distances[(station, target)] = dist_map[target]
    
    # Sort tasks by end time for DP
    sorted_tasks = sorted(tasks, key=lambda x: x['end'])
    n = len(sorted_tasks)
    
    # Fast non-overlapping predecessor lookup
    def find_latest_non_overlapping(i):
        for j in range(i - 1, -1, -1):
            if sorted_tasks[j]['end'] <= sorted_tasks[i]['start']:
                return j
        return -1
    
    # DP with single pass
    # dp[i] = (max_score, min_cost, schedule) for best solution using tasks 0..i
    dp = {}
    
    # Base case
    first_task = sorted_tasks[0]
    cost = distances[(start_station, first_task['station'])] + distances[(first_task['station'], start_station)]
    dp[-1] = (0, 0, [])  # Empty solution
    dp[0] = (first_task['score'], cost, [first_task['name']])
    
    for i in range(1, n):
        current_task = sorted_tasks[i]
        
        # Option 1: Don't take current task
        dp[i] = dp[i-1]
        
        # Option 2: Take current task
        latest_compatible = find_latest_non_overlapping(i)
        prev_score, prev_cost, prev_schedule = dp[latest_compatible]
        
        if latest_compatible == -1:
            # Start fresh
            new_cost = distances[(start_station, current_task['station'])] + distances[(current_task['station'], start_station)]
            new_score = current_task['score']
            new_schedule = [current_task['name']]
        else:
            # Extend previous solution
            if prev_schedule:
                # Find the station of the last task in previous schedule
                last_task_name = prev_schedule[-1]
                last_station = next(t['station'] for t in sorted_tasks if t['name'] == last_task_name)
                
                # Update cost: remove old return, add travel to current, add new return
                old_return_cost = distances[(last_station, start_station)]
                travel_cost = distances[(last_station, current_task['station'])]
                new_return_cost = distances[(current_task['station'], start_station)]
                
                new_cost = prev_cost - old_return_cost + travel_cost + new_return_cost
            else:
                new_cost = distances[(start_station, current_task['station'])] + distances[(current_task['station'], start_station)]
            
            new_score = prev_score + current_task['score']
            new_schedule = prev_schedule + [current_task['name']]
        
        # Keep the better solution
        current_score, current_cost, current_schedule = dp[i]
        if (new_score > current_score or 
            (new_score == current_score and new_cost < current_cost)):
            dp[i] = (new_score, new_cost, new_schedule)
    
    max_score, min_cost, best_schedule = dp[n-1]
    
    # Sort schedule by start time
    if best_schedule:
        task_start_map = {task['name']: task['start'] for task in tasks}
        best_schedule.sort(key=lambda name: task_start_map[name])
    
    return max_score, min_cost, best_schedule

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
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)