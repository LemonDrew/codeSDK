from flask import Flask, request, jsonify
from collections import defaultdict

from routes import app

def find_cycle_edges(connections):
    graph = defaultdict(list)
    for c in connections:
        graph[c["spy1"]].append(c["spy2"])
        graph[c["spy2"]].append(c["spy1"])

    visited = set()
    parent = {}
    cycle_edges = []

    def dfs(u, p):
        visited.add(u)
        for v in graph[u]:
            if v == p:
                continue
            if v in visited:
                # Found a back edge (u, v)
                if (v, u) not in cycle_edges:  # avoid duplicates
                    cycle_edges.append((u, v))
            else:
                parent[v] = u
                dfs(v, u)

    for node in graph:
        if node not in visited:
            dfs(node, None)

    # Convert to list of dicts
    return [{"spy1": u, "spy2": v} for u, v in cycle_edges]

@app.route("/investigate", methods=["POST"])
def investigate():
    data = request.get_json()
    networks = data.get("networks", [])

    results = []
    for net in networks:
        network_id = net.get("networkId")
        connections = net.get("network", [])

        extra_channels = find_cycle_edges(connections)

        results.append({
            "networkId": network_id,
            "extraChannels": extra_channels
        })

    return jsonify({"networks": results})

if __name__ == "__main__":
    app.run(port=3000, debug=True)
