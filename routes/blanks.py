import json
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
import logging
from flask import request, jsonify
from routes import app

logger = logging.getLogger(__name__)

class ConservativeImputer:
    """
    Conservative imputation focused on minimizing error rather than fitting complex patterns
    """
    
    def _safe_array_conversion(self, series):
        """Safely convert series to numpy array"""
        try:
            converted = []
            for val in series:
                if val is None:
                    converted.append(np.nan)
                else:
                    try:
                        num_val = float(val)
                        if np.isfinite(num_val):
                            converted.append(num_val)
                        else:
                            converted.append(np.nan)
                    except (ValueError, TypeError, OverflowError):
                        converted.append(np.nan)
            
            return np.array(converted, dtype=np.float64)
            
        except Exception:
            return np.full(len(series), np.nan, dtype=np.float64)
    
    def _extract_valid_data(self, arr):
        """Extract valid data points"""
        try:
            valid_mask = np.isfinite(arr)
            valid_indices = np.where(valid_mask)[0]
            valid_values = arr[valid_mask]
            
            if len(valid_values) > 0:
                finite_mask = np.isfinite(valid_values)
                valid_indices = valid_indices[finite_mask]
                valid_values = valid_values[finite_mask]
            
            return valid_indices, valid_values, valid_mask
            
        except Exception:
            return np.array([]), np.array([]), np.zeros(len(arr), dtype=bool)
    
    def _estimate_local_trend(self, indices, values, target_idx, window_size=10):
        """Estimate local trend around a target index"""
        try:
            if len(indices) < 2:
                return values[0] if len(values) > 0 else 0.0
            
            # Find points within window
            distances = np.abs(indices - target_idx)
            sorted_idx = np.argsort(distances)
            
            # Use up to window_size nearest points
            local_count = min(window_size, len(indices))
            local_indices = indices[sorted_idx[:local_count]]
            local_values = values[sorted_idx[:local_count]]
            
            if len(local_values) == 1:
                return local_values[0]
            elif len(local_values) == 2:
                # Linear interpolation between two points
                x1, x2 = local_indices[0], local_indices[1]
                y1, y2 = local_values[0], local_values[1]
                if x2 != x1:
                    return y1 + (y2 - y1) * (target_idx - x1) / (x2 - x1)
                else:
                    return y1
            else:
                # Local linear regression
                coeffs = np.polyfit(local_indices, local_values, 1)
                return np.polyval(coeffs, target_idx)
                
        except Exception:
            return np.mean(values) if len(values) > 0 else 0.0
    
    def _detect_simple_pattern(self, indices, values):
        """Very conservative pattern detection"""
        if len(values) < 5:
            return 'local_linear'
        
        try:
            # Only detect very clear patterns
            
            # Test for strong linear trend
            linear_coeffs = np.polyfit(indices, values, 1)
            linear_pred = np.polyval(linear_coeffs, indices)
            linear_r2 = 1 - np.var(values - linear_pred) / np.var(values) if np.var(values) > 1e-10 else 0.9
            
            # Test for strong quadratic trend (only if much better than linear)
            if len(values) >= 8:
                quad_coeffs = np.polyfit(indices, values, 2)
                quad_pred = np.polyval(quad_coeffs, indices)
                quad_r2 = 1 - np.var(values - quad_pred) / np.var(values) if np.var(values) > 1e-10 else 0.9
                
                # Only use quadratic if it's significantly better AND has high R²
                if quad_r2 > 0.95 and quad_r2 > linear_r2 + 0.05:
                    return 'quadratic'
            
            # Strong linear trend
            if linear_r2 > 0.9:
                return 'linear'
            
            # Default to local linear (safest)
            return 'local_linear'
            
        except Exception:
            return 'local_linear'
    
    def _local_linear_interpolation(self, series):
        """Local linear interpolation - most conservative approach"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) == 0:
            return np.zeros(len(series))
        elif len(values) == 1:
            return np.full(len(series), values[0])
        
        result = np.full(len(series), np.nan)
        
        # Fill known values
        result[indices] = values
        
        # Fill missing values using local trends
        for i in range(len(series)):
            if np.isnan(result[i]):
                result[i] = self._estimate_local_trend(indices, values, i, window_size=6)
        
        # Final cleanup
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        return result
    
    def _conservative_linear_interpolation(self, series):
        """Standard linear interpolation with conservative extrapolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) == 0:
            return np.zeros(len(series))
        elif len(values) == 1:
            return np.full(len(series), values[0])
        
        try:
            f = interpolate.interp1d(
                indices, values,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            
            result = f(np.arange(len(series)))
            
            # Conservative extrapolation bounds
            data_range = np.ptp(values)
            if data_range > 0:
                center = np.median(values)
                max_deviation = 2 * data_range  # Very conservative
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._local_linear_interpolation(series)
    
    def _conservative_quadratic_interpolation(self, series):
        """Conservative quadratic interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 4:
            return self._conservative_linear_interpolation(series)
        
        try:
            coeffs = np.polyfit(indices, values, 2)
            result = np.polyval(coeffs, np.arange(len(series)))
            
            # Very conservative extrapolation bounds
            data_range = np.ptp(values)
            if data_range > 0:
                center = np.median(values)
                max_deviation = 1.5 * data_range  # Even more conservative
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._conservative_linear_interpolation(series)
    
    def _conservative_spline_interpolation(self, series):
        """Conservative cubic spline with heavy regularization"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 5:
            return self._conservative_linear_interpolation(series)
        
        try:
            # Sort data
            sort_idx = np.argsort(indices)
            sorted_indices = indices[sort_idx]
            sorted_values = values[sort_idx]
            
            # Use natural boundary conditions for stability
            cs = interpolate.CubicSpline(
                sorted_indices, sorted_values,
                bc_type='natural',
                extrapolate=True
            )
            
            result = cs(np.arange(len(series)))
            
            # Very conservative bounds to prevent overfitting
            data_range = np.ptp(sorted_values)
            if data_range > 0:
                center = np.median(sorted_values)
                max_deviation = 2 * data_range
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._conservative_linear_interpolation(series)
    
    def _smoothed_interpolation(self, series):
        """Smoothed interpolation for noisy data"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 8:
            return self._conservative_linear_interpolation(series)
        
        try:
            # Apply light smoothing first
            if len(values) >= 7:
                window_length = min(7, len(values))
                if window_length % 2 == 0:
                    window_length -= 1
                
                try:
                    smoothed_values = savgol_filter(values, window_length, 2)
                except Exception:
                    smoothed_values = values
            else:
                smoothed_values = values
            
            # Linear interpolation on smoothed data
            f = interpolate.interp1d(
                indices, smoothed_values,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            
            result = f(np.arange(len(series)))
            
            # Conservative bounds
            data_range = np.ptp(values)  # Use original values for bounds
            if data_range > 0:
                center = np.median(values)
                max_deviation = 2 * data_range
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._conservative_linear_interpolation(series)
    
    def impute_series(self, series):
        """Main imputation with very conservative approach"""
        try:
            if not series or len(series) == 0:
                return []
            
            arr = self._safe_array_conversion(series)
            indices, values, valid_mask = self._extract_valid_data(arr)
            
            # Handle edge cases
            if len(values) == 0:
                return [0.0] * len(series)
            elif len(values) == len(series):
                return [float(x) if x is not None else 0.0 for x in series]
            elif len(values) == 1:
                return [float(values[0])] * len(series)
            elif len(values) == 2:
                return self._conservative_linear_interpolation(series).tolist()
            
            # Conservative pattern detection
            pattern = self._detect_simple_pattern(indices, values)
            
            # Apply most appropriate method
            if pattern == 'quadratic':
                result = self._conservative_quadratic_interpolation(series)
            elif pattern == 'linear':
                result = self._conservative_linear_interpolation(series)
            else:  # local_linear (default)
                # Choose between local linear and smoothed based on data characteristics
                if len(values) >= 10:
                    # For longer series, try smoothed interpolation
                    result = self._smoothed_interpolation(series)
                else:
                    # For shorter series, use local linear
                    result = self._local_linear_interpolation(series)
            
            # Final validation and conversion
            result = np.asarray(result, dtype=np.float64)
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return [float(x) for x in result]
            
        except Exception as e:
            logger.error(f"Imputation error: {str(e)}")
            # Ultimate safe fallback
            valid_vals = [x for x in series if x is not None]
            if valid_vals:
                mean_val = sum(valid_vals) / len(valid_vals)
                return [float(x) if x is not None else float(mean_val) for x in series]
            else:
                return [0.0] * len(series)

# Initialize conservative imputer
imputer = ConservativeImputer()

@app.route('/blankety', methods=['POST'])
def blankety_blanks():
    try:
        data = request.get_json()
        
        if not data or 'series' not in data:
            return jsonify({'error': 'Missing series data'}), 400
        
        series_list = data['series']
        
        if not isinstance(series_list, list):
            return jsonify({'error': 'Series must be a list'}), 400
        
        completed_series = []
        
        for series in series_list:
            if not isinstance(series, list):
                return jsonify({'error': 'Each series must be a list'}), 400
            
            completed = imputer.impute_series(series)
            completed_series.append(completed)
        
        return jsonify({'answer': completed_series})
        
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500