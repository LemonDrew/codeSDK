from flask import Flask, request, jsonify

from routes import app

def merge_intervals(intervals):
    """
    Merge overlapping intervals and return sorted merged intervals.
    
    Args:
        intervals: List of [start, end] intervals
    
    Returns:
        List of merged intervals sorted by start time
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last_merged = merged[-1]
        
        # If current interval overlaps with the last merged interval
        if current[0] <= last_merged[1]:
            # Merge by extending the end time
            merged[-1] = [last_merged[0], max(last_merged[1], current[1])]
        else:
            # No overlap, add current interval
            merged.append(current)
    
    return merged

def min_boats_needed(intervals):
    """
    Find the minimum number of boats needed to satisfy all bookings.
    This is equivalent to finding the maximum number of overlapping intervals.
    
    Args:
        intervals: List of [start, end] intervals
    
    Returns:
        Integer representing minimum boats needed
    """
    if not intervals:
        return 0
    
    # Create events: +1 for start, -1 for end
    events = []
    for start, end in intervals:
        events.append((start, 1))    # Booking starts
        events.append((end, -1))     # Booking ends
    
    # Sort events by time, with ends before starts for same time
    events.sort(key=lambda x: (x[0], x[1]))
    
    max_boats = 0
    current_boats = 0
    
    for time, event_type in events:
        current_boats += event_type
        max_boats = max(max_boats, current_boats)
    
    return max_boats

@app.route('/sailing-club/submission', methods=['POST'])
def sailingclub():
    try:
        data = request.get_json()
        
        if not data or 'testCases' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        solutions = []
        
        for test_case in data['testCases']:
            if 'id' not in test_case or 'input' not in test_case:
                return jsonify({'error': f'Missing id or input in test case'}), 400
            
            test_id = test_case['id']
            intervals = test_case['input']
            
            # Part 1: Merge overlapping intervals
            sorted_merged_slots = merge_intervals(intervals)
            
            # Part 2: Find minimum boats needed
            min_boats = min_boats_needed(intervals)
            
            solution = {
                'id': test_id,
                'sortedMergedSlots': sorted_merged_slots,
                'minBoatsNeeded': min_boats
            }
            
            solutions.append(solution)
        
        return jsonify({'solutions': solutions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
