from flask import Flask, request, jsonify
from routes import app

@app.route("/princess-diaries", methods=["POST"])
def princess_diaries():
    data = request.get_json()
    tasks = data["tasks"]
    subway = data["subway"]
    start_station = data["starting_station"]
    
    stations = set()
    adjacency = {}
    
    for route in subway:
        u, v = route["connection"]
        fee = route["fee"]
        stations.update([u, v])
        
        if u not in adjacency:
            adjacency[u] = {}
        if v not in adjacency:
            adjacency[v] = {}
            
        adjacency[u][v] = min(adjacency[u].get(v, float('inf')), fee)
        adjacency[v][u] = min(adjacency[v].get(u, float('inf')), fee)
    
    stations = list(stations)
    V = len(stations)
    station_idx = {s: i for i, s in enumerate(stations)}

    INF = float('inf')
    dist = [[INF if i != j else 0 for j in range(V)] for i in range(V)]
    
    for station, neighbors in adjacency.items():
        i = station_idx[station]
        for neighbor, fee in neighbors.items():
            j = station_idx[neighbor]
            dist[i][j] = fee
    
    for k in range(V):
        for i in range(V):
            if dist[i][k] != INF:  # Skip if no path to k
                for j in range(V):
                    if dist[k][j] != INF:  # Skip if no path from k
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    

    tasks.sort(key=lambda x: x["end"])
    
    task_stations = [station_idx[task["station"]] for task in tasks]
    start_idx = station_idx[start_station]
    
    N = len(tasks)
    dp = [(0, INF, [])] * N  # (score, fee, schedule)
    
    for i in range(N):
        task = tasks[i]
        task_station_idx = task_stations[i]
        
        # Base case: just this task
        best_score = task["score"]
        best_fee = dist[start_idx][task_station_idx]
        best_schedule = [task["name"]]
        
        # Try combining with previous non-overlapping tasks
        for j in range(i):
            if tasks[j]["end"] <= task["start"]:  # Non-overlapping
                prev_score, prev_fee, prev_schedule = dp[j]
                if prev_fee != INF:  # Valid previous state
                    travel_fee = dist[task_stations[j]][task_station_idx]
                    if travel_fee != INF:  # Valid travel path
                        score = prev_score + task["score"]
                        fee = prev_fee + travel_fee
                        
                        # Update if better score, or same score with lower fee
                        if (score > best_score or 
                            (score == best_score and fee < best_fee)):
                            best_score = score
                            best_fee = fee
                            best_schedule = prev_schedule + [task["name"]]
        
        dp[i] = (best_score, best_fee, best_schedule)
    
    # Find the best final result
    best_result = max(dp, key=lambda x: (x[0], -x[1]) if x[1] != INF else (-1, 0))
    max_score, min_fee, schedule = best_result
    
    # Add return trip cost
    if schedule and min_fee != INF:
        # Find the last task's station
        last_task_name = schedule[-1]
        last_task_station = None
        for task in tasks:
            if task["name"] == last_task_name:
                last_task_station = station_idx[task["station"]]
                break
        
        if last_task_station is not None:
            return_cost = dist[last_task_station][start_idx]
            if return_cost != INF:
                min_fee += return_cost
    
    # Handle case where no valid solution exists
    if min_fee == INF:
        min_fee = 0
        max_score = 0
        schedule = []
    
    return jsonify({
        "max_score": max_score,
        "min_fee": min_fee,
        "schedule": schedule
    })

if __name__ == "__main__":
    app.run(port=3000, debug=True)