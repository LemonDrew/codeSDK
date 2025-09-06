from routes import app
from flask import Flask, request, jsonify

@app.route('/sailing-club/submission', methods=['POST'])
def sailing_club():
    data = request.get_json()
    test_cases = data.get("testCases")

    result = []

    for test_case in test_cases:
        id = test_case["id"]
        schedule = test_case["input"]

        sorted_input = sort_input(schedule)

        result.append({
            "id" : id, 
            "sortedMergedSlots": sorted_input,
            "minBoatsNeeded?": 2
        })

    return jsonify(result)


def sort_input(schedule):

    schedule.sort(key = lambda x : x[0]) # Sort by starting date

    result = []
    current = []

    for interval in schedule:
        start_time, end_time = interval      

        if current == []:
            current = [start_time, end_time]
            continue

        if current[0] <= start_time <= current[1]:
            current = [current[0], end_time]
            continue
        
        else:
            result.append(current)
            current = [start_time, end_time]
            continue

    result.append(current)

    return result



