from flask import Flask, request, jsonify

app = Flask(__name__)

# Utility: Union-Find (Disjoint Set Union)
class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX == rootY:
            return False  
        self.parent[rootY] = rootX
        return True


@app.route("/investigate", methods=["POST"])
def investigate():
    data = request.get_json()
    networks = data.get("networks", [])

    results = []
    for net in networks:
        network_id = net.get("networkId")
        connections = net.get("network", [])
        
        dsu = DSU()
        extra_channels = []

        for connection in connections:
            spy1 = connection["spy1"]
            spy2 = connection["spy2"]
            if not dsu.union(spy1, spy2):
                # edge creates a cycle
                extra_channels.append({"spy1": spy1, "spy2": spy2})

        results.append({
            "networkId": network_id,
            "extraChannels": extra_channels
        })

    return jsonify({"networks": results})


if __name__ == "__main__":
    app.run(port=3000, debug=True)
