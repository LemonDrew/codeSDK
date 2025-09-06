import random
from flask import Flask, request, jsonify

from routes import app

@app.route("/trading-bot", methods=["POST"])
def bot():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not isinstance(data, list):
        return jsonify({"error": "Body must be a JSON array of news events"}), 400

    # Collect unique IDs from input
    ids = []
    seen = set()
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            # Skip malformed entries
            continue
        _id = item["id"]
        if _id not in seen:
            seen.add(_id)
            ids.append(_id)

    if not ids:
        return jsonify([]), 200

    # Pick up to 50 unique IDs
    k = min(50, len(ids))
    selected_ids = random.sample(ids, k)

    # Random decisions (intentionally ignoring any market logic)
    decisions = [{"id": _id, "decision": random.choice(["LONG", "SHORT"])}
                 for _id in selected_ids]

    return jsonify(decisions), 200
