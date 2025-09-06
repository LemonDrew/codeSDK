from flask import Flask, jsonify

from routes import app

@app.route('/mst-calculation', methods=['POST'])
def mst():
    data = [
        {"value": 9},
        {"value": 8}
    ]
    return jsonify(data)