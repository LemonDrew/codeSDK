import json
import logging
import math
import statistics

from flask import request, jsonify

from routes import app

logger = logging.getLogger(__name__)

def safe_mean(values):
    """Safe mean calculation"""
    if not values:
        return 0.0
    return sum(values) / len(values)

def simple_linear_impute(series):
    """Ultra-simple linear interpolation"""
    # Find valid points
    valid = [(i, val) for i, val in enumerate(series) if val is not None]
    
    if not valid:
        return [0.0] * len(series)
    if len(valid) == 1:
        return [valid[0][1]] * len(series)
    
    result = list(series)
    
    # Fill missing values
    for i in range(len(series)):
        if series[i] is None:
            # Find neighbors
            left = None
            right = None
            
            for idx, val in valid:
                if idx < i:
                    left = (idx, val)
                elif idx > i and right is None:
                    right = (idx, val)
                    break
            
            if left and right:
                # Interpolate
                x1, y1 = left
                x2, y2 = right
                t = (i - x1) / (x2 - x1)
                result[i] = y1 + t * (y2 - y1)
            elif left:
                # Extrapolate from trend
                if len(valid) >= 2:
                    p1_idx, p1_val = valid[-2]
                    p2_idx, p2_val = valid[-1]
                    if p2_idx != p1_idx:
                        slope = (p2_val - p1_val) / (p2_idx - p1_idx)
                        result[i] = p2_val + slope * (i - p2_idx)
                    else:
                        result[i] = left[1]
                else:
                    result[i] = left[1]
            elif right:
                result[i] = right[1]
            else:
                result[i] = 0.0
    
    return result

@app.route('/blankety', methods=['POST'])
def blankety_blanks():
    try:
        data = request.get_json()
        
        if not data or 'series' not in data:
            return jsonify({'error': 'Missing series data'}), 400
        
        series_list = data['series']
        completed_series = []
        
        for series in series_list:
            completed = simple_linear_impute(series)
            completed_series.append(completed)
        
        return jsonify({'answer': completed_series})
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500