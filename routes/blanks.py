import json
import logging
import math
import statistics

from flask import request, jsonify

from routes import app

logger = logging.getLogger(__name__)

class FastImputer:
    """
    Fast imputation using heuristic method selection instead of cross-validation.
    Picks methods based on data characteristics, not expensive testing.
    """
    
    def _get_valid_points(self, series):
        """Extract valid data points with their indices"""
        valid_points = []
        for i, val in enumerate(series):
            if val is not None:
                valid_points.append((i, val))
        return valid_points
    
    def _analyze_pattern(self, valid_points):
        """
        Fast pattern analysis using simple heuristics
        Returns: pattern_type, confidence
        """
        if len(valid_points) < 3:
            return 'linear', 1.0
        
        values = [val for idx, val in valid_points]
        indices = [idx for idx, val in valid_points]
        
        # Calculate basic statistics
        value_range = max(values) - min(values)
        mean_val = statistics.mean(values)
        
        # Check for exponential pattern (rapid growth/decay)
        if all(v > 0 for v in values) and value_range > abs(mean_val):
            ratios = []
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    ratios.append(values[i] / values[i-1])
            
            if ratios and statistics.stdev(ratios) < 0.5:  # Consistent ratio
                avg_ratio = statistics.mean(ratios)
                if avg_ratio > 1.2 or avg_ratio < 0.8:  # Significant growth/decay
                    return 'exponential', 0.8
        
        # Check for periodic pattern (simple peak detection)
        if len(valid_points) >= 10:
            # Look for repeating patterns in differences
            diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
            if len(diffs) >= 6:
                # Check if pattern repeats every 5, 10, or 20 points
                for period in [5, 10, 20]:
                    if len(diffs) >= period * 2:
                        correlations = []
                        for i in range(len(diffs) - period):
                            correlations.append(diffs[i] * diffs[i + period])
                        
                        if correlations and statistics.mean(correlations) > 0:
                            return 'seasonal', 0.6
        
        # Check for quadratic pattern (acceleration/deceleration)
        if len(valid_points) >= 5:
            # Calculate second differences
            first_diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
            second_diffs = [first_diffs[i+1] - first_diffs[i] for i in range(len(first_diffs)-1)]
            
            if second_diffs:
                avg_second_diff = statistics.mean([abs(d) for d in second_diffs])
                if avg_second_diff > value_range * 0.1:  # Significant curvature
                    return 'quadratic', 0.7
        
        # Default to linear for steady trends
        return 'linear', 0.9
    
    def _linear_interpolation(self, series):
        """Fast linear interpolation"""
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 2:
            if valid_points:
                return [valid_points[0][1]] * len(series)
            return [0.0] * len(series)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # Linear interpolation/extrapolation
        for i in range(len(series)):
            if result[i] is None:
                # Find nearest neighbors
                left_point = None
                right_point = None
                
                for idx, val in valid_points:
                    if idx < i:
                        left_point = (idx, val)
                    elif idx > i and right_point is None:
                        right_point = (idx, val)
                        break
                
                if left_point and right_point:
                    # Interpolate
                    x1, y1 = left_point
                    x2, y2 = right_point
                    t = (i - x1) / (x2 - x1)
                    result[i] = y1 + t * (y2 - y1)
                elif left_point:
                    # Extrapolate forward
                    if len(valid_points) >= 2:
                        x1, y1 = valid_points[-2]
                        x2, y2 = valid_points[-1]
                        slope = (y2 - y1) / (x2 - x1)
                        result[i] = y2 + slope * (i - x2)
                    else:
                        result[i] = left_point[1]
                elif right_point:
                    # Extrapolate backward  
                    if len(valid_points) >= 2:
                        x1, y1 = valid_points[0]
                        x2, y2 = valid_points[1]
                        slope = (y2 - y1) / (x2 - x1)
                        result[i] = y1 + slope * (i - x1)
                    else:
                        result[i] = right_point[1]
        
        return result
    
    def _quadratic_interpolation(self, series):
        """Fast quadratic interpolation using local fits"""
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 3:
            return self._linear_interpolation(series)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # Quadratic interpolation
        for i in range(len(series)):
            if result[i] is None:
                # Find best triplet around position i
                best_triplet = None
                min_distance = float('inf')
                
                for j in range(len(valid_points) - 2):
                    triplet = valid_points[j:j+3]
                    center_idx = triplet[1][0]
                    distance = abs(center_idx - i)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_triplet = triplet
                
                if best_triplet:
                    # Lagrange interpolation
                    (x1, y1), (x2, y2), (x3, y3) = best_triplet
                    x = i
                    
                    term1 = y1 * ((x - x2) * (x - x3)) / ((x1 - x2) * (x1 - x3))
                    term2 = y2 * ((x - x1) * (x - x3)) / ((x2 - x1) * (x2 - x3))
                    term3 = y3 * ((x - x1) * (x - x2)) / ((x3 - x1) * (x3 - x2))
                    
                    result[i] = term1 + term2 + term3
                else:
                    # Fallback to linear
                    linear_result = self._linear_interpolation(series)
                    result[i] = linear_result[i]
        
        return result
    
    def _seasonal_interpolation(self, series):
        """Fast seasonal interpolation using detected period"""
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 10:
            return self._linear_interpolation(series)
        
        # Quick period detection
        period = self._detect_period_fast(valid_points)
        if not period:
            return self._linear_interpolation(series)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # Use periodic pattern
        for i in range(len(series)):
            if result[i] is None:
                # Look for values at same phase in other cycles
                candidates = []
                
                for offset in [-period, period, -2*period, 2*period]:
                    ref_idx = i + offset
                    if 0 <= ref_idx < len(series):
                        for idx, val in valid_points:
                            if idx == ref_idx:
                                candidates.append(val)
                                break
                
                if candidates:
                    result[i] = statistics.mean(candidates)
                else:
                    # Fallback to linear
                    linear_result = self._linear_interpolation(series)
                    result[i] = linear_result[i]
        
        return result
    
    def _exponential_interpolation(self, series):
        """Fast exponential interpolation"""
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 3:
            return self._linear_interpolation(series)
        
        # Check if all values are positive
        values = [val for idx, val in valid_points]
        if min(values) <= 0:
            return self._quadratic_interpolation(series)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        try:
            # Simple exponential fit using first and last points
            x1, y1 = valid_points[0]
            x2, y2 = valid_points[-1]
            
            if y1 > 0 and y2 > 0:
                # Calculate growth rate
                growth_rate = (y2 / y1) ** (1.0 / (x2 - x1))
                
                # Fill missing values
                for i in range(len(series)):
                    if result[i] is None:
                        result[i] = y1 * (growth_rate ** (i - x1))
            else:
                return self._quadratic_interpolation(series)
                
        except (ValueError, OverflowError, ZeroDivisionError):
            return self._quadratic_interpolation(series)
        
        return result
    
    def _detect_period_fast(self, valid_points):
        """Fast period detection using autocorrelation"""
        if len(valid_points) < 20:
            return None
        
        values = [val for idx, val in valid_points]
        
        # Test common periods
        for period in [5, 10, 20, 50]:
            if period >= len(values) // 2:
                continue
            
            # Simple correlation check
            correlations = []
            for i in range(len(values) - period):
                correlations.append(values[i] * values[i + period])
            
            if correlations:
                mean_corr = statistics.mean(correlations)
                if mean_corr > 0.5:  # Simple threshold
                    return period
        
        return None
    
    def impute_series(self, series):
        """Main imputation with fast heuristic method selection"""
        valid_points = self._get_valid_points(series)
        
        # Handle edge cases
        if len(valid_points) == 0:
            return [0.0] * len(series)
        elif len(valid_points) == len(series):
            return series
        elif len(valid_points) == 1:
            return [valid_points[0][1]] * len(series)
        
        # Fast pattern analysis
        pattern_type, confidence = self._analyze_pattern(valid_points)
        
        # Apply appropriate method based on pattern
        try:
            if pattern_type == 'exponential' and confidence > 0.7:
                result = self._exponential_interpolation(series)
            elif pattern_type == 'seasonal' and confidence > 0.5:
                result = self._seasonal_interpolation(series)
            elif pattern_type == 'quadratic' and confidence > 0.6:
                result = self._quadratic_interpolation(series)
            else:
                result = self._linear_interpolation(series)
            
            # Ensure no None values
            for i in range(len(result)):
                if result[i] is None:
                    # Final fallback
                    if valid_points:
                        result[i] = statistics.mean([val for idx, val in valid_points])
                    else:
                        result[i] = 0.0
            
            return result
            
        except Exception:
            # Ultimate fallback
            return self._linear_interpolation(series)

# Initialize imputer
imputer = FastImputer()

@app.route('/blankety', methods=['POST'])
def blankety_blanks():
    try:
        data = request.get_json()
        
        if not data or 'series' not in data:
            return jsonify({'error': 'Missing series data'}), 400
        
        series_list = data['series']
        
        # REMOVED: Strict validation that was causing errors
        # Accept any number of series and any length
        
        completed_series = []
        
        for series in series_list:
            # Impute the series (whatever length it is)
            completed = imputer.impute_series(series)
            completed_series.append(completed)
        
        return jsonify({'answer': completed_series})
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500