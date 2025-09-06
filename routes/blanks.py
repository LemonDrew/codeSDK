import json
import logging

from flask import request, jsonify

import numpy as np
from scipy import interpolate

from routes import app

logger = logging.getLogger(__name__)

def impute_series(series):
    arr = np.array([x if x is not None else np.nan for x in series])
    valid_mask = ~np.isnan(arr)
    valid_indices = np.where(valid_mask)[0]
    valid_values = arr[valid_mask]

    if len(valid_values) == 0:
        return [0.0] * len(series)
    elif len(valid_values) == 1:
        return [valid_values[0]] * len(series)
    elif len(valid_values) == 2:
        # Linear
        slope = (valid_values[1] - valid_values[0]) / (valid_indices[1] - valid_indices[0])
        result = []
        for i in range(len(series)):
            value = valid_values[0] + slope * (i - valid_indices[0])
            result.append(value)
        return result
    try:
        # Create cubic spline interpolator
        cs = interpolate.CubicSpline(valid_indices, valid_values, 
                                   bc_type='natural', extrapolate=True)
        
        # Generate all indices and interpolate
        all_indices = np.arange(len(series))
        interpolated = cs(all_indices)
        
        # Replace only the missing values, keep original valid ones
        result = arr.copy()
        result[~valid_mask] = interpolated[~valid_mask]
        
        return result.tolist()
    except Exception:
        # Fallback
        f = interpolate.interp1d(valid_indices, valid_values, 
                               kind='linear', fill_value='extrapolate')
        return f(np.arange(len(series))).tolist()
    



@app.route('/blankety', methods=['POST'])
def blankety_blanks():
    try:
        data = request.get_json()
        series_list = data.get("series")
        completed_series = []
        for series in series_list:
            completed = impute_series(series)
            completed_series.append(completed)
        return jsonify({"completed_series": completed_series})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    