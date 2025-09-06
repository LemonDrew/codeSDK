from flask import Flask, request, jsonify
import itertools
from routes import app

@app.route("/princess-diaries", methods=["POST"])
def princess_diaries():
    data = request.get_json()
    tasks = data["tasks"]
    subway = data["subway"]
    start_station = data["starting_station"]

  
    stations = set()
    for route in subway:
        stations.update(route["connection"])
    stations = list(stations)
    V = len(stations)
    station_idx = {s: i for i, s in enumerate(stations)}

  
    INF = 1 << 60
    dist = [[INF] * V for _ in range(V)]
    for i in range(V):
        dist[i][i] = 0
    for route in subway:
        u, v = route["connection"]
        u_idx, v_idx = station_idx[u], station_idx[v]
        dist[u_idx][v_idx] = min(dist[u_idx][v_idx], route["fee"])
        dist[v_idx][u_idx] = min(dist[v_idx][u_idx], route["fee"])

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    tasks.sort(key=lambda x: x["end"])

    N = len(tasks)
    dp = [None] * N  # Each element: (score, fee, schedule)
    for i, t in enumerate(tasks):
        best_score = t["score"]
        best_fee = dist[station_idx[start_station]][station_idx[t["station"]]]
        best_schedule = [t["name"]]

        for j in range(i):
            t_prev = tasks[j]
            if t_prev["end"] <= t["start"]:  # Non-overlapping
                prev_score, prev_fee, prev_schedule = dp[j]
                travel_fee = dist[station_idx[t_prev["station"]]][station_idx[t["station"]]]
                score = prev_score + t["score"]
                fee = prev_fee + travel_fee
                if score > best_score or (score == best_score and fee < best_fee):
                    best_score = score
                    best_fee = fee
                    best_schedule = prev_schedule + [t["name"]]

        dp[i] = (best_score, best_fee, best_schedule)


    max_score, min_fee, schedule = max(dp, key=lambda x: (x[0], -x[1]))


    last_station = tasks[[t["name"] for t in tasks].index(schedule[-1])]["station"]
    min_fee += dist[station_idx[last_station]][station_idx[start_station]]

    return jsonify({
        "max_score": max_score,
        "min_fee": min_fee,
        "schedule": schedule
    })

if __name__ == "__main__":
    app.run(port=3000, debug=True)
