import json

import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
import logging

from flask import request, jsonify

from routes import app

logger = logging.getLogger(__name__)

class MathSafeImputer:
    """
    Mathematically safe imputation with comprehensive error prevention
    """
    
    def _safe_array_conversion(self, series):
        """Safely convert series to numpy array with validation"""
        try:
            # Convert None to NaN, ensure all values are numeric
            converted = []
            for val in series:
                if val is None:
                    converted.append(np.nan)
                else:
                    try:
                        num_val = float(val)
                        # Check for problematic values
                        if np.isfinite(num_val):
                            converted.append(num_val)
                        else:
                            converted.append(np.nan)
                    except (ValueError, TypeError, OverflowError):
                        converted.append(np.nan)
            
            arr = np.array(converted, dtype=np.float64)
            return arr
            
        except Exception:
            # Ultimate fallback - return array of NaNs
            return np.full(len(series), np.nan, dtype=np.float64)
    
    def _extract_valid_data(self, arr):
        """Extract valid indices and values with safety checks"""
        try:
            valid_mask = np.isfinite(arr)
            valid_indices = np.where(valid_mask)[0]
            valid_values = arr[valid_mask]
            
            # Additional safety - remove any remaining problematic values
            if len(valid_values) > 0:
                finite_mask = np.isfinite(valid_values)
                valid_indices = valid_indices[finite_mask]
                valid_values = valid_values[finite_mask]
            
            return valid_indices, valid_values, valid_mask
            
        except Exception:
            return np.array([]), np.array([]), np.zeros(len(arr), dtype=bool)
    
    def _safe_polyfit(self, x, y, degree):
        """Polynomial fitting with overflow protection"""
        try:
            if len(x) <= degree or len(y) <= degree:
                return None
            
            # Normalize x values to prevent numerical issues
            x_min, x_max = np.min(x), np.max(x)
            if x_max == x_min:
                return None
            
            x_norm = (x - x_min) / (x_max - x_min)
            
            # Fit polynomial on normalized data
            coeffs_norm = np.polyfit(x_norm, y, degree)
            
            # Convert coefficients back to original scale
            # This is complex for higher degrees, so we'll use the normalized form
            return coeffs_norm, x_min, x_max
            
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return None
    
    def _safe_polyval(self, coeffs_data, x):
        """Polynomial evaluation with overflow protection"""
        try:
            if coeffs_data is None:
                return None
            
            coeffs_norm, x_min, x_max = coeffs_data
            
            # Normalize x values
            if x_max == x_min:
                x_norm = np.zeros_like(x)
            else:
                x_norm = (x - x_min) / (x_max - x_min)
            
            # Evaluate polynomial
            result = np.polyval(coeffs_norm, x_norm)
            
            # Clamp results to prevent extreme values
            if len(result) > 0:
                result = np.clip(result, -1e10, 1e10)
            
            return result
            
        except (ValueError, OverflowError):
            return None
    
    def _detect_pattern_safe(self, indices, values):
        """Safe pattern detection with comprehensive error handling"""
        if len(values) < 3:
            return 'linear'
        
        try:
            # Check for exponential pattern (positive values only)
            if len(values) >= 4 and np.all(values > 0):
                try:
                    log_values = np.log(values)
                    if np.all(np.isfinite(log_values)):
                        # Linear fit in log space
                        linear_fit = self._safe_polyfit(indices, log_values, 1)
                        if linear_fit is not None:
                            pred_log = self._safe_polyval(linear_fit, indices)
                            if pred_log is not None:
                                residuals = log_values - pred_log
                                if np.var(log_values) > 1e-10:  # Avoid division by zero
                                    r_squared = 1 - np.var(residuals) / np.var(log_values)
                                    if r_squared > 0.8:
                                        return 'exponential'
                except (ValueError, OverflowError):
                    pass
            
            # Check for quadratic pattern
            if len(values) >= 5:
                try:
                    quad_fit = self._safe_polyfit(indices, values, 2)
                    linear_fit = self._safe_polyfit(indices, values, 1)
                    
                    if quad_fit is not None and linear_fit is not None:
                        quad_pred = self._safe_polyval(quad_fit, indices)
                        linear_pred = self._safe_polyval(linear_fit, indices)
                        
                        if quad_pred is not None and linear_pred is not None:
                            if np.var(values) > 1e-10:  # Avoid division by zero
                                quad_r2 = 1 - np.var(values - quad_pred) / np.var(values)
                                linear_r2 = 1 - np.var(values - linear_pred) / np.var(values)
                                
                                if quad_r2 > linear_r2 + 0.1 and quad_r2 > 0.7:
                                    return 'quadratic'
                except (ValueError, OverflowError):
                    pass
            
            # Check for periodic pattern using simple autocorrelation
            if len(values) >= 20:
                try:
                    # Remove linear trend first
                    trend_fit = self._safe_polyfit(indices, values, 1)
                    if trend_fit is not None:
                        trend = self._safe_polyval(trend_fit, indices)
                        if trend is not None:
                            detrended = values - trend
                            
                            # Simple periodicity check
                            for period in [5, 10, 20, 50]:
                                if period < len(detrended) // 2:
                                    correlations = []
                                    for i in range(len(detrended) - period):
                                        correlations.append(detrended[i] * detrended[i + period])
                                    
                                    if len(correlations) > 0:
                                        mean_corr = np.mean(correlations)
                                        if mean_corr > 0.5:
                                            return 'periodic'
                except (ValueError, OverflowError):
                    pass
            
            return 'linear'
            
        except Exception:
            return 'linear'
    
    def _linear_interpolation_safe(self, series):
        """Ultra-safe linear interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) == 0:
            return np.zeros(len(series), dtype=np.float64)
        elif len(values) == 1:
            return np.full(len(series), values[0], dtype=np.float64)
        
        try:
            # Use scipy's robust interpolation
            f = interpolate.interp1d(
                indices, values, 
                kind='linear', 
                fill_value='extrapolate',
                bounds_error=False,
                assume_sorted=False
            )
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = f(all_indices)
            
            # Safety checks and cleanup
            result = np.asarray(result, dtype=np.float64)
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return result
            
        except Exception:
            # Manual linear interpolation fallback
            result = np.full(len(series), np.mean(values), dtype=np.float64)
            return result
    
    def _cubic_spline_safe(self, series):
        """Safe cubic spline interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 4:
            return self._linear_interpolation_safe(series)
        
        try:
            # Ensure indices are sorted
            sort_idx = np.argsort(indices)
            sorted_indices = indices[sort_idx]
            sorted_values = values[sort_idx]
            
            # Create cubic spline
            cs = interpolate.CubicSpline(
                sorted_indices, sorted_values,
                bc_type='natural',
                extrapolate=True
            )
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = cs(all_indices)
            
            # Prevent extreme extrapolation
            value_range = np.ptp(sorted_values)
            if value_range > 0:
                center = np.median(sorted_values)
                max_deviation = 5 * value_range
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._linear_interpolation_safe(series)
    
    def _exponential_safe(self, series):
        """Safe exponential interpolation with strict bounds"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 3 or not np.all(values > 0):
            return self._cubic_spline_safe(series)
        
        try:
            # Fit in log space with safety checks
            log_values = np.log(np.maximum(values, 1e-10))  # Prevent log(0)
            
            if not np.all(np.isfinite(log_values)):
                return self._cubic_spline_safe(series)
            
            # Linear fit in log space
            coeffs = np.polyfit(indices, log_values, 1)
            slope, intercept = coeffs
            
            # Limit exponential growth to prevent overflow
            slope = np.clip(slope, -0.01, 0.01)
            
            # Generate result
            all_indices = np.arange(len(series), dtype=np.float64)
            log_result = slope * all_indices + intercept
            
            # Prevent overflow in exponential
            log_result = np.clip(log_result, -20, 20)
            result = np.exp(log_result)
            
            # Additional bounds based on original data
            min_val = np.min(values)
            max_val = np.max(values)
            result = np.clip(result, min_val * 0.1, max_val * 10)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._cubic_spline_safe(series)
    
    def _quadratic_safe(self, series):
        """Safe quadratic interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 3:
            return self._linear_interpolation_safe(series)
        
        try:
            # Fit quadratic polynomial
            degree = min(2, len(values) - 1)
            coeffs = np.polyfit(indices, values, degree)
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = np.polyval(coeffs, all_indices)
            
            # Prevent extreme extrapolation
            value_range = np.ptp(values)
            if value_range > 0:
                center = np.median(values)
                max_deviation = 3 * value_range
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._linear_interpolation_safe(series)
    
    def _periodic_safe(self, series):
        """Safe periodic interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 10:
            return self._cubic_spline_safe(series)
        
        try:
            # Remove trend
            trend_coeffs = np.polyfit(indices, values, 1)
            trend = np.polyval(trend_coeffs, indices)
            detrended = values - trend
            
            # Apply smoothing if enough data
            if len(detrended) >= 7:
                window_length = min(len(detrended) // 4, 15)
                window_length = max(5, window_length)
                if window_length % 2 == 0:
                    window_length += 1
                
                try:
                    smoothed = savgol_filter(detrended, window_length, 3)
                except Exception:
                    smoothed = detrended
            else:
                smoothed = detrended
            
            # Interpolate smoothed data
            f = interpolate.interp1d(
                indices, smoothed,
                kind='cubic',
                fill_value='extrapolate',
                bounds_error=False
            )
            
            all_indices = np.arange(len(series), dtype=np.float64)
            full_trend = np.polyval(trend_coeffs, all_indices)
            periodic_part = f(all_indices)
            
            result = full_trend + periodic_part
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return result
            
        except Exception:
            return self._cubic_spline_safe(series)
    
    def impute_series(self, series):
        """Main imputation with comprehensive safety"""
        try:
            # Validate input
            if not series or len(series) == 0:
                return []
            
            arr = self._safe_array_conversion(series)
            indices, values, valid_mask = self._extract_valid_data(arr)
            
            # Handle edge cases
            if len(values) == 0:
                return [0.0] * len(series)
            elif len(values) == len(series):
                # No missing values, just clean the data
                return [float(x) if x is not None else 0.0 for x in series]
            elif len(values) == 1:
                return [float(values[0])] * len(series)
            
            # Detect pattern safely
            pattern = self._detect_pattern_safe(indices, values)
            
            # Apply appropriate method
            if pattern == 'exponential':
                result = self._exponential_safe(series)
            elif pattern == 'quadratic':
                result = self._quadratic_safe(series)
            elif pattern == 'periodic':
                result = self._periodic_safe(series)
            else:  # linear
                result = self._cubic_spline_safe(series)
            
            # Final validation
            result = np.asarray(result, dtype=np.float64)
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Convert to regular Python list
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

# Initialize imputer
imputer = MathSafeImputer()

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