from flask import Flask, request, jsonify
from collections import defaultdict

from routes import app

def has_cycle_dfs(graph, all_nodes):
    """Check if the graph has a cycle using DFS"""
    visited = set()
    rec_stack = set()
    
    def dfs(node, parent):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            
            if neighbor in rec_stack:
                return True
            
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
        
        rec_stack.remove(node)
        return False
    
    # Check each connected component
    for node in all_nodes:
        if node not in visited:
            if dfs(node, None):
                return True
    
    return False

def find_all_cycle_edges(connections):
    if not connections:
        return []
    
    # Build adjacency list and collect all nodes
    graph = defaultdict(set)
    all_nodes = set()
    
    for conn in connections:
        spy1, spy2 = conn["spy1"], conn["spy2"]
        graph[spy1].add(spy2)
        graph[spy2].add(spy1)
        all_nodes.add(spy1)
        all_nodes.add(spy2)
    
    # Check if the original graph has any cycles
    if not has_cycle_dfs(graph, all_nodes):
        return []  # No cycles, so no cycle edges
    
    cycle_edges = []
    
    # Try removing each edge and check if cycles still exist
    for conn in connections:
        spy1, spy2 = conn["spy1"], conn["spy2"]
        
        # Create a copy of the graph without this edge
        temp_graph = defaultdict(set)
        for node in graph:
            temp_graph[node] = graph[node].copy()
        
        # Remove the edge
        temp_graph[spy1].discard(spy2)
        temp_graph[spy2].discard(spy1)
        
        # If removing this edge eliminates all cycles, then this edge was part of a cycle
        if not has_cycle_dfs(temp_graph, all_nodes):
            cycle_edges.append({"spy1": spy1, "spy2": spy2})
    
    return cycle_edges

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