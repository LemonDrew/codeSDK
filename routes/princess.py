from flask import Flask, request, jsonify
import heapq
from collections import defaultdict
import sys

from routes import app

def dijkstra(graph, start, end):
    """Find shortest path cost between start and end using Dijkstra's algorithm"""
    if start == end:
        return 0
    
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == end:
            return current_dist
            
        for neighbor, weight in graph[current]:
            if neighbor not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return float('inf')

def build_graph(subway_routes):
    """Build adjacency list representation of the subway system"""
    graph = defaultdict(list)
    for route in subway_routes:
        station1, station2 = route['connection']
        fee = route['fee']
        graph[station1].append((station2, fee))
        graph[station2].append((station1, fee))
    return graph

def tasks_overlap(task1, task2):
    """Check if two tasks have overlapping time windows"""
    return not (task1['end'] <= task2['start'] or task2['end'] <= task1['start'])

def find_optimal_schedule(tasks, subway_routes, starting_station):
    """Find the schedule that maximizes score and minimizes transport cost"""
    
    # Build subway graph
    graph = build_graph(subway_routes)
    
    # Sort tasks by start time
    sorted_tasks = sorted(tasks, key=lambda x: x['start'])
    n = len(sorted_tasks)
    
    # Precompute all shortest distances
    all_stations = set([starting_station])
    for task in tasks:
        all_stations.add(task['station'])
    
    distances = {}
    for station1 in all_stations:
        for station2 in all_stations:
            if station1 != station2:
                distances[(station1, station2)] = dijkstra(graph, station1, station2)
            else:
                distances[(station1, station2)] = 0
    
    # Dynamic programming to find maximum score schedules
    # dp[i] = list of (score, min_cost, schedule) for schedules ending at task i
    dp = [[] for _ in range(n)]
    
    # Base case: each task can be done alone
    for i in range(n):
        task = sorted_tasks[i]
        cost = distances[(starting_station, task['station'])] + distances[(task['station'], starting_station)]
        dp[i].append((task['score'], cost, [task['name']]))
    
    # Fill DP table
    for i in range(n):
        current_task = sorted_tasks[i]
        
        # Try extending previous schedules
        for j in range(i):
            prev_task = sorted_tasks[j]
            
            # Check if tasks don't overlap
            if not tasks_overlap(current_task, prev_task):
                # Extend each schedule ending at j
                for prev_score, prev_cost, prev_schedule in dp[j]:
                    # Calculate additional cost to go from prev_task to current_task
                    additional_cost = distances[(prev_task['station'], current_task['station'])]
                    # Cost to return to starting station from current task
                    return_cost = distances[(current_task['station'], starting_station)]
                    
                    new_score = prev_score + current_task['score']
                    # Remove the return cost from previous schedule and add new costs
                    new_cost = prev_cost - distances[(prev_task['station'], starting_station)] + additional_cost + return_cost
                    new_schedule = prev_schedule + [current_task['name']]
                    
                    dp[i].append((new_score, new_cost, new_schedule))
    
    # Find the schedule with maximum score and minimum cost
    best_schedules = []
    max_score = 0
    
    # Collect all schedules from all ending positions
    all_schedules = []
    for schedules in dp:
        all_schedules.extend(schedules)
    
    # Find maximum score
    for score, cost, schedule in all_schedules:
        if score > max_score:
            max_score = score
    
    # Find minimum cost among maximum score schedules
    min_cost = float('inf')
    for score, cost, schedule in all_schedules:
        if score == max_score and cost < min_cost:
            min_cost = cost
            best_schedules = [schedule]
        elif score == max_score and cost == min_cost:
            best_schedules.append(schedule)
    
    if not best_schedules:
        return 0, 0, []
    
    # Return the first optimal schedule (they're all equivalent)
    return max_score, min_cost, best_schedules[0]

@app.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    try:
        data = request.get_json()
        
        tasks = data['tasks']
        subway_routes = data['subway']
        starting_station = data['starting_station']
        
        max_score, min_fee, schedule = find_optimal_schedule(tasks, subway_routes, starting_station)
        
        # Sort schedule by start time (should already be sorted from our algorithm)
        task_start_times = {task['name']: task['start'] for task in tasks}
        schedule.sort(key=lambda name: task_start_times[name])
        
        return jsonify({
            "max_score": max_score,
            "min_fee": min_fee,
            "schedule": schedule
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500