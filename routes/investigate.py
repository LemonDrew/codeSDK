from flask import Flask, request, jsonify
from collections import defaultdict

from routes import app

def find_all_cycle_edges(connections):
    if not connections:
        return []
    
    # Build adjacency list
    graph = defaultdict(set)
    edge_set = set()
    
    for conn in connections:
        spy1, spy2 = conn["spy1"], conn["spy2"]
        graph[spy1].add(spy2)
        graph[spy2].add(spy1)
        # Store edges in a normalized form to avoid duplicates
        edge_set.add((min(spy1, spy2), max(spy1, spy2)))
    
    # Find all edges that are part of cycles
    cycle_edges = set()
    
    def dfs_find_cycles(node, parent, visited, rec_stack, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
                
            if neighbor in rec_stack:
                # Found a cycle - add all edges in the cycle
                cycle_start_idx = path.index(neighbor)
                cycle_path = path[cycle_start_idx:] + [neighbor]
                
                # Add all edges in this cycle
                for i in range(len(cycle_path) - 1):
                    u, v = cycle_path[i], cycle_path[i + 1]
                    cycle_edges.add((min(u, v), max(u, v)))
            
            elif neighbor not in visited:
                dfs_find_cycles(neighbor, node, visited, rec_stack, path)
        
        path.pop()
        rec_stack.remove(node)
    
    visited = set()
    all_nodes = set()
    for conn in connections:
        all_nodes.add(conn["spy1"])
        all_nodes.add(conn["spy2"])
    
    for node in all_nodes:
        if node not in visited:
            dfs_find_cycles(node, None, visited, set(), [])
    
    # Convert back to the required format
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