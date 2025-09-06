from flask import Flask, request, jsonify
from collections import defaultdict

from routes import app

def find_all_cycle_edges(connections):
    if not connections:
        return []
    
    # Build adjacency list
    graph = defaultdict(list)
    for conn in connections:
        spy1, spy2 = conn["spy1"], conn["spy2"]
        graph[spy1].append(spy2)
        graph[spy2].append(spy1)
    
    visited = set()
    cycle_edges = set()
    
    def dfs(node, parent, path):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            
            if neighbor in path:
                # Found a cycle - mark all edges in the cycle
                cycle_start = path.index(neighbor)
                cycle_nodes = path[cycle_start:] + [neighbor]
                
                for i in range(len(cycle_nodes) - 1):
                    u, v = cycle_nodes[i], cycle_nodes[i + 1]
                    cycle_edges.add((min(u, v), max(u, v)))
            
            elif neighbor not in visited:
                dfs(neighbor, node, path + [node])
    
    # Get all nodes
    all_nodes = set()
    for conn in connections:
        all_nodes.add(conn["spy1"])
        all_nodes.add(conn["spy2"])
    
    # Start DFS from each unvisited node
    for node in all_nodes:
        if node not in visited:
            dfs(node, None, [])
    
    # Return edges that are part of cycles, preserving original format
    result = []
    for conn in connections:
        spy1, spy2 = conn["spy1"], conn["spy2"]
        edge = (min(spy1, spy2), max(spy1, spy2))
        if edge in cycle_edges:
            result.append({"spy1": spy1, "spy2": spy2})
    
    return result

@app.route("/investigate", methods=["POST"])
def investigate():
    data = request.get_json()
    
    # Handle both possible input formats
    if isinstance(data, dict) and "networks" in data:
        networks = data["networks"]
    elif isinstance(data, list):
        networks = data

    results = []
    for net in networks:
        network_id = net.get("networkId")
        connections = net.get("network", [])

        extra_channels = find_all_cycle_edges(connections)

        results.append({
            "networkId": network_id,
            "extraChannels": extra_channels
        })

    return jsonify({"networks": results})

if __name__ == "__main__":
    app.run(port=3000, debug=True)