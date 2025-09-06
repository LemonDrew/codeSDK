from flask import Flask, request, jsonify
import json
import math
import statistics

app = Flask(__name__)

class MarketSignalImputer:
    """
    - Linear: trends and steady growth
    - Quadratic: curved trends and acceleration
    - Moving Average: smooth local patterns
    - Seasonal: periodic/cyclical patterns
    - Exponential: exponential growth/decay
    """
    
    def __init__(self):
        self.methods = {
            'linear': self._linear_interpolation,
            'quadratic': self._quadratic_interpolation, 
            'moving_average': self._moving_average_interpolation,
            'seasonal': self._seasonal_interpolation,
            'exponential': self._exponential_interpolation
        }
    
    def _get_valid_points(self, series):
        """Extract valid data points with their indices"""
        valid_points = []
        for i, val in enumerate(series):
            if val is not None:
                valid_points.append((i, val))
        return valid_points
    
    def _linear_interpolation(self, series):
        """
        Linear interpolation - best for steady trends
        Uses slope between nearest neighbors
        """
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 2:
            return self._fallback_fill(series, valid_points)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # Interpolate missing values
        for i in range(len(series)):
            if result[i] is None:
                # Find surrounding points
                left_point = None
                right_point = None
                
                for idx, val in valid_points:
                    if idx < i:
                        left_point = (idx, val)
                    elif idx > i and right_point is None:
                        right_point = (idx, val)
                        break
                
                if left_point and right_point:
                    # Linear interpolation
                    x1, y1 = left_point
                    x2, y2 = right_point
                    t = (i - x1) / (x2 - x1)
                    result[i] = y1 + t * (y2 - y1)
                elif left_point:
                    # Extrapolate forward using last trend
                    result[i] = self._extrapolate_linear(valid_points, i, 'forward')
                elif right_point:
                    # Extrapolate backward using first trend
                    result[i] = self._extrapolate_linear(valid_points, i, 'backward')
        
        return result
    
    def _quadratic_interpolation(self, series):
        """
        Quadratic interpolation - captures curved trends and acceleration
        Fits parabolas through local triplets of points
        """
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 3:
            return self._linear_interpolation(series)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # Interpolate missing values
        for i in range(len(series)):
            if result[i] is None:
                # Find best triplet of points around position i
                best_triplet = self._find_best_triplet(valid_points, i)
                if best_triplet:
                    result[i] = self._quadratic_fit(best_triplet, i)
                else:
                    # Fall back to linear
                    result[i] = self._linear_interpolation(series)[i]
        
        return result
    
    def _moving_average_interpolation(self, series):
        """
        Moving average based interpolation - smooths local variations
        Uses weighted average of nearby points
        """
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 3:
            return self._linear_interpolation(series)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # First pass: linear interpolation
        linear_result = self._linear_interpolation(series)
        
        # Second pass: smooth with moving average
        window_size = min(20, len(valid_points) // 3)
        
        for i in range(len(series)):
            if result[i] is None:
                # Find nearby points within window
                nearby_points = []
                for idx, val in valid_points:
                    if abs(idx - i) <= window_size:
                        weight = 1.0 / (1.0 + abs(idx - i))  # Distance-based weight
                        nearby_points.append((val, weight))
                
                if nearby_points:
                    # Weighted average
                    weighted_sum = sum(val * weight for val, weight in nearby_points)
                    total_weight = sum(weight for val, weight in nearby_points)
                    result[i] = weighted_sum / total_weight
                else:
                    result[i] = linear_result[i]
        
        return result
    
    def _seasonal_interpolation(self, series):
        """
        Seasonal/periodic interpolation - detects and uses repeating patterns
        Looks for periodic behavior to predict missing values
        """
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 10:
            return self._linear_interpolation(series)
        
        # Detect potential periods (common market cycles)
        potential_periods = [5, 10, 20, 50, 100, 250]  # Daily, weekly, monthly patterns
        best_period = self._detect_best_period(valid_points, potential_periods)
        
        if best_period:
            return self._periodic_interpolation(series, valid_points, best_period)
        else:
            return self._moving_average_interpolation(series)
    
    def _exponential_interpolation(self, series):
        """
        Exponential interpolation - for exponential growth/decay patterns
        Fits exponential curves to capture compound growth
        """
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 3:
            return self._linear_interpolation(series)
        
        # Check if data might be exponential (all positive and varying significantly)
        values = [val for idx, val in valid_points]
        if min(values) <= 0 or max(values) / min(values) < 2:
            return self._quadratic_interpolation(series)
        
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # Log-transform for exponential fitting
        try:
            log_points = [(idx, math.log(val)) for idx, val in valid_points if val > 0]
            if len(log_points) < 2:
                return self._quadratic_interpolation(series)
            
            # Fit line in log space
            for i in range(len(series)):
                if result[i] is None:
                    log_value = self._interpolate_log_linear(log_points, i)
                    result[i] = math.exp(log_value)
        
        except (ValueError, OverflowError):
            return self._quadratic_interpolation(series)
        
        return result
    
    def _find_best_triplet(self, valid_points, target_idx):
        """Find the best triplet of points for quadratic fitting"""
        if len(valid_points) < 3:
            return None
        
        # Find triplet that best surrounds the target index
        best_triplet = None
        min_span = float('inf')
        
        for i in range(len(valid_points) - 2):
            triplet = valid_points[i:i+3]
            indices = [idx for idx, val in triplet]
            
            # Check if target is within or close to the span
            span = max(indices) - min(indices)
            if min(indices) <= target_idx <= max(indices):
                if span < min_span:
                    min_span = span
                    best_triplet = triplet
        
        # If no containing triplet, find closest one
        if not best_triplet:
            distances = []
            for i in range(len(valid_points) - 2):
                triplet = valid_points[i:i+3]
                center_idx = triplet[1][0]  # Middle point index
                distance = abs(center_idx - target_idx)
                distances.append((distance, triplet))
            
            best_triplet = min(distances)[1]
        
        return best_triplet
    
    def _quadratic_fit(self, triplet, target_idx):
        """Fit quadratic through three points and evaluate at target"""
        (x1, y1), (x2, y2), (x3, y3) = triplet
        
        # Solve for quadratic coefficients: y = ax² + bx + c
        # Using Lagrange interpolation for stability
        x = target_idx
        
        term1 = y1 * ((x - x2) * (x - x3)) / ((x1 - x2) * (x1 - x3))
        term2 = y2 * ((x - x1) * (x - x3)) / ((x2 - x1) * (x2 - x3))
        term3 = y3 * ((x - x1) * (x - x2)) / ((x3 - x1) * (x3 - x2))
        
        return term1 + term2 + term3
    
    def _detect_best_period(self, valid_points, potential_periods):
        """Detect the best periodic pattern in the data"""
        if len(valid_points) < 20:
            return None
        
        best_period = None
        best_score = -1
        
        for period in potential_periods:
            if period >= len(valid_points) // 3:
                continue
            
            score = self._calculate_periodicity_score(valid_points, period)
            if score > best_score:
                best_score = score
                best_period = period
        
        return best_period if best_score > 0.3 else None
    
    def _calculate_periodicity_score(self, valid_points, period):
        """Calculate how well the data fits a given period"""
        correlations = []
        
        for i in range(len(valid_points) - period):
            if i + period < len(valid_points):
                val1 = valid_points[i][1]
                val2 = valid_points[i + period][1]
                correlations.append(val1 * val2)
        
        if not correlations:
            return 0
        
        # Simple correlation measure
        mean_corr = statistics.mean(correlations)
        return min(1.0, max(0.0, mean_corr / (statistics.stdev(correlations) + 1e-6)))
    
    def _periodic_interpolation(self, series, valid_points, period):
        """Interpolate using detected periodic pattern"""
        result = [None] * len(series)
        
        # Fill known values
        for idx, val in valid_points:
            result[idx] = val
        
        # Use periodic pattern for interpolation
        for i in range(len(series)):
            if result[i] is None:
                # Find corresponding point in previous/next cycles
                cycle_values = []
                
                for offset in [-period, period, -2*period, 2*period]:
                    ref_idx = i + offset
                    if 0 <= ref_idx < len(series):
                        for idx, val in valid_points:
                            if idx == ref_idx:
                                cycle_values.append(val)
                                break
                
                if cycle_values:
                    result[i] = statistics.mean(cycle_values)
                else:
                    # Fall back to linear interpolation
                    linear_result = self._linear_interpolation(series)
                    result[i] = linear_result[i]
        
        return result
    
    def _interpolate_log_linear(self, log_points, target_idx):
        """Linear interpolation in log space"""
        if len(log_points) < 2:
            return 0
        
        # Find surrounding points
        left_point = None
        right_point = None
        
        for idx, log_val in log_points:
            if idx < target_idx:
                left_point = (idx, log_val)
            elif idx > target_idx and right_point is None:
                right_point = (idx, log_val)
                break
        
        if left_point and right_point:
            x1, y1 = left_point
            x2, y2 = right_point
            t = (target_idx - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
        elif left_point:
            return left_point[1]
        elif right_point:
            return right_point[1]
        else:
            return statistics.mean([log_val for idx, log_val in log_points])
    
    def _extrapolate_linear(self, valid_points, target_idx, direction):
        """Extrapolate using linear trend"""
        if len(valid_points) < 2:
            return valid_points[0][1] if valid_points else 0
        
        if direction == 'forward':
            # Use last two points
            (x1, y1), (x2, y2) = valid_points[-2:]
        else:
            # Use first two points
            (x1, y1), (x2, y2) = valid_points[:2]
        
        slope = (y2 - y1) / (x2 - x1)
        return y2 + slope * (target_idx - x2)
    
    def _fallback_fill(self, series, valid_points):
        """Simple fallback for edge cases"""
        if not valid_points:
            return [0.0] * len(series)
        elif len(valid_points) == 1:
            return [valid_points[0][1]] * len(series)
        else:
            return self._linear_interpolation(series)
    
    def _evaluate_method(self, series, method_name):
        """
        Evaluate method quality using cross-validation
        Hides some known points and measures prediction accuracy
        """
        valid_points = self._get_valid_points(series)
        if len(valid_points) < 10:
            return float('inf')
        
        # Hide 20% of points for testing
        test_size = max(1, len(valid_points) // 5)
        test_indices = set(range(0, len(valid_points), max(1, len(valid_points) // test_size)))
        
        # Create test series with some points hidden
        test_series = series.copy()
        true_values = {}
        
        for i, (idx, val) in enumerate(valid_points):
            if i in test_indices:
                test_series[idx] = None
                true_values[idx] = val
        
        # Impute and calculate error
        try:
            method = self.methods[method_name]
            imputed = method(test_series)
            
            errors = []
            for idx, true_val in true_values.items():
                pred_val = imputed[idx]
                if pred_val is not None:
                    errors.append(abs(pred_val - true_val))
            
            return statistics.mean(errors) if errors else float('inf')
            
        except Exception:
            return float('inf')
    
    def impute_series(self, series):
        """
        Main imputation method - tries multiple approaches and picks the best
        """
        valid_points = self._get_valid_points(series)
        
        # Handle trivial cases
        if len(valid_points) == 0:
            return [0.0] * len(series)
        elif len(valid_points) == len(series):
            return series  # No missing values
        
        # For small datasets, use simple linear interpolation
        if len(valid_points) < 5:
            return self._linear_interpolation(series)
        
        # Evaluate different methods (sample-based for speed)
        method_errors = {}
        
        # Always test linear (fast and reliable)
        method_errors['linear'] = self._evaluate_method(series, 'linear')
        
        # Test other methods based on data characteristics
        values = [val for idx, val in valid_points]
        value_range = max(values) - min(values)
        
        # Test quadratic for curved patterns
        if value_range > 0.1:  # Only if there's significant variation
            method_errors['quadratic'] = self._evaluate_method(series, 'quadratic')
        
        # Test moving average for noisy data
        if len(valid_points) >= 10:
            method_errors['moving_average'] = self._evaluate_method(series, 'moving_average')
        
        # Test seasonal for longer series
        if len(valid_points) >= 20:
            method_errors['seasonal'] = self._evaluate_method(series, 'seasonal')
        
        # Test exponential for growth patterns
        if all(v > 0 for v in values) and max(values) / min(values) > 2:
            method_errors['exponential'] = self._evaluate_method(series, 'exponential')
        
        # Pick best method
        best_method = min(method_errors, key=method_errors.get)
        
        # Apply best method
        try:
            result = self.methods[best_method](series)
            
            # Ensure no None values remain
            for i in range(len(result)):
                if result[i] is None:
                    # Use linear interpolation as final fallback
                    linear_result = self._linear_interpolation(series)
                    result[i] = linear_result[i] if linear_result[i] is not None else 0.0
            
            return result
            
        except Exception:
            # Ultimate fallback
            return self._linear_interpolation(series)

# Initialize the imputer
imputer = MarketSignalImputer()

@app.route('/blankety', methods=['POST'])
def blankety_blanks():
    try:
        data = request.get_json()
        
        if not data or 'series' not in data:
            return jsonify({'error': 'Missing series data'}), 400
        
        series_list = data['series']
        
        if not isinstance(series_list, list) or len(series_list) != 100:
            return jsonify({'error': 'Expected exactly 100 series'}), 400
        
        completed_series = []
        method_stats = {}
        
        for i, series in enumerate(series_list):
            if not isinstance(series, list) or len(series) != 1000:
                return jsonify({'error': f'Series {i} must have exactly 1000 elements'}), 400
            
            # Impute the series
            completed = imputer.impute_series(series)
            completed_series.append(completed)
        
        return jsonify({'answer': completed_series})
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500