import json
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import logging
from flask import request, jsonify
from routes import app

logger = logging.getLogger(__name__)

class EnhancedImputer:
    """
    Enhanced imputation with better pattern detection and accuracy
    """
    
    def _safe_array_conversion(self, series):
        """Safely convert series to numpy array with validation"""
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
            
            arr = np.array(converted, dtype=np.float64)
            return arr
            
        except Exception:
            return np.full(len(series), np.nan, dtype=np.float64)
    
    def _extract_valid_data(self, arr):
        """Extract valid indices and values with safety checks"""
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
    
    def _detect_noise_level(self, indices, values):
        """Estimate noise level in the data"""
        if len(values) < 5:
            return 0.1
        
        try:
            # Fit linear trend and measure residuals
            coeffs = np.polyfit(indices, values, 1)
            trend = np.polyval(coeffs, indices)
            residuals = values - trend
            noise_std = np.std(residuals)
            
            # Normalize by data range
            data_range = np.ptp(values)
            if data_range > 0:
                noise_level = noise_std / data_range
            else:
                noise_level = 0.1
            
            return min(noise_level, 0.5)  # Cap at 50%
            
        except Exception:
            return 0.1
    
    def _enhanced_pattern_detection(self, indices, values):
        """Enhanced pattern detection with multiple criteria"""
        if len(values) < 3:
            return 'linear', 0.5
        
        try:
            patterns = {}
            
            # Test linear fit
            try:
                linear_coeffs = np.polyfit(indices, values, 1)
                linear_pred = np.polyval(linear_coeffs, indices)
                linear_mse = np.mean((values - linear_pred) ** 2)
                data_var = np.var(values)
                if data_var > 1e-10:
                    linear_r2 = 1 - linear_mse / data_var
                    patterns['linear'] = max(0, linear_r2)
                else:
                    patterns['linear'] = 0.9
            except Exception:
                patterns['linear'] = 0.5
            
            # Test quadratic fit
            if len(values) >= 4:
                try:
                    quad_coeffs = np.polyfit(indices, values, 2)
                    quad_pred = np.polyval(quad_coeffs, indices)
                    quad_mse = np.mean((values - quad_pred) ** 2)
                    if data_var > 1e-10:
                        quad_r2 = 1 - quad_mse / data_var
                        patterns['quadratic'] = max(0, quad_r2)
                    else:
                        patterns['quadratic'] = 0.9
                except Exception:
                    patterns['quadratic'] = 0.0
            
            # Test exponential fit (positive values only)
            if len(values) >= 4 and np.all(values > 0):
                try:
                    log_values = np.log(values)
                    if np.all(np.isfinite(log_values)):
                        exp_coeffs = np.polyfit(indices, log_values, 1)
                        exp_pred_log = np.polyval(exp_coeffs, indices)
                        exp_mse = np.mean((log_values - exp_pred_log) ** 2)
                        log_var = np.var(log_values)
                        if log_var > 1e-10:
                            exp_r2 = 1 - exp_mse / log_var
                            patterns['exponential'] = max(0, exp_r2)
                        else:
                            patterns['exponential'] = 0.9
                except Exception:
                    patterns['exponential'] = 0.0
            
            # Test periodic pattern
            if len(values) >= 15:
                try:
                    # Remove linear trend first
                    detrended = values - np.polyval(np.polyfit(indices, values, 1), indices)
                    
                    # Test multiple periods
                    periodic_scores = []
                    for period in range(3, min(len(values) // 3, 50)):
                        if len(detrended) >= 2 * period:
                            autocorr = 0
                            count = 0
                            for i in range(len(detrended) - period):
                                autocorr += detrended[i] * detrended[i + period]
                                count += 1
                            
                            if count > 0:
                                autocorr /= count
                                periodic_scores.append(autocorr)
                    
                    if periodic_scores:
                        max_autocorr = max(periodic_scores)
                        avg_autocorr = np.mean(periodic_scores)
                        if avg_autocorr != 0:
                            periodic_strength = max_autocorr / (abs(avg_autocorr) + 1e-10)
                            patterns['periodic'] = min(0.9, max(0, periodic_strength - 1) * 0.5)
                        else:
                            patterns['periodic'] = 0.0
                    else:
                        patterns['periodic'] = 0.0
                        
                except Exception:
                    patterns['periodic'] = 0.0
            
            # Select best pattern
            if not patterns:
                return 'linear', 0.5
            
            best_pattern = max(patterns, key=patterns.get)
            confidence = patterns[best_pattern]
            
            # Require minimum confidence for non-linear patterns
            if best_pattern == 'exponential' and confidence < 0.85:
                best_pattern = 'quadratic' if patterns.get('quadratic', 0) > 0.7 else 'linear'
            elif best_pattern == 'periodic' and confidence < 0.6:
                best_pattern = 'quadratic' if patterns.get('quadratic', 0) > 0.7 else 'linear'
            elif best_pattern == 'quadratic' and confidence < 0.75:
                best_pattern = 'linear'
            
            return best_pattern, patterns.get(best_pattern, 0.5)
            
        except Exception:
            return 'linear', 0.5
    
    def _akima_interpolation(self, series):
        """Akima spline interpolation - good for avoiding oscillations"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 4:
            return self._linear_interpolation_safe(series)
        
        try:
            # Sort data
            sort_idx = np.argsort(indices)
            sorted_indices = indices[sort_idx]
            sorted_values = values[sort_idx]
            
            # Use Akima interpolation
            akima = interpolate.Akima1DInterpolator(sorted_indices, sorted_values)
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = akima(all_indices, extrapolate=True)
            
            # Prevent extreme extrapolation
            value_range = np.ptp(sorted_values)
            if value_range > 0:
                center = np.median(sorted_values)
                max_deviation = 4 * value_range
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._linear_interpolation_safe(series)
    
    def _pchip_interpolation(self, series):
        """PCHIP interpolation - preserves monotonicity"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 3:
            return self._linear_interpolation_safe(series)
        
        try:
            # Sort data
            sort_idx = np.argsort(indices)
            sorted_indices = indices[sort_idx]
            sorted_values = values[sort_idx]
            
            # Use PCHIP interpolation
            pchip = interpolate.PchipInterpolator(sorted_indices, sorted_values, extrapolate=True)
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = pchip(all_indices)
            
            # Gentle bounds to preserve shape
            value_range = np.ptp(sorted_values)
            if value_range > 0:
                center = np.median(sorted_values)
                max_deviation = 6 * value_range
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._linear_interpolation_safe(series)
    
    def _smart_spline_selection(self, series):
        """Intelligently select between different spline types"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 4:
            return self._linear_interpolation_safe(series)
        
        try:
            noise_level = self._detect_noise_level(indices, values)
            
            # For low noise data, use cubic spline
            if noise_level < 0.05:
                return self._cubic_spline_safe(series)
            # For medium noise, use Akima
            elif noise_level < 0.15:
                return self._akima_interpolation(series)
            # For high noise, use PCHIP or smoothed cubic
            else:
                return self._pchip_interpolation(series)
                
        except Exception:
            return self._cubic_spline_safe(series)
    
    def _linear_interpolation_safe(self, series):
        """Enhanced linear interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) == 0:
            return np.zeros(len(series), dtype=np.float64)
        elif len(values) == 1:
            return np.full(len(series), values[0], dtype=np.float64)
        
        try:
            f = interpolate.interp1d(
                indices, values, 
                kind='linear', 
                fill_value='extrapolate',
                bounds_error=False,
                assume_sorted=False
            )
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = f(all_indices)
            
            result = np.asarray(result, dtype=np.float64)
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return result
            
        except Exception:
            result = np.full(len(series), np.mean(values), dtype=np.float64)
            return result
    
    def _cubic_spline_safe(self, series):
        """Enhanced cubic spline with better boundary conditions"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 4:
            return self._linear_interpolation_safe(series)
        
        try:
            sort_idx = np.argsort(indices)
            sorted_indices = indices[sort_idx]
            sorted_values = values[sort_idx]
            
            # Choose boundary condition based on data characteristics
            noise_level = self._detect_noise_level(indices, values)
            if noise_level < 0.1:
                bc_type = 'not-a-knot'  # Better for smooth data
            else:
                bc_type = 'natural'     # More stable for noisy data
            
            cs = interpolate.CubicSpline(
                sorted_indices, sorted_values,
                bc_type=bc_type,
                extrapolate=True
            )
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = cs(all_indices)
            
            # Conservative extrapolation bounds
            value_range = np.ptp(sorted_values)
            if value_range > 0:
                center = np.median(sorted_values)
                max_deviation = 3 * value_range
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._linear_interpolation_safe(series)
    
    def _exponential_safe(self, series):
        """More conservative exponential interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 4 or not np.all(values > 0):
            return self._smart_spline_selection(series)
        
        try:
            log_values = np.log(np.maximum(values, 1e-10))
            
            if not np.all(np.isfinite(log_values)):
                return self._smart_spline_selection(series)
            
            # More conservative slope limits
            coeffs = np.polyfit(indices, log_values, 1)
            slope, intercept = coeffs
            slope = np.clip(slope, -0.005, 0.005)  # Even more conservative
            
            all_indices = np.arange(len(series), dtype=np.float64)
            log_result = slope * all_indices + intercept
            log_result = np.clip(log_result, -15, 15)  # Tighter bounds
            result = np.exp(log_result)
            
            # Tighter bounds based on original data
            min_val = np.min(values)
            max_val = np.max(values)
            result = np.clip(result, min_val * 0.2, max_val * 5)  # More conservative
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._smart_spline_selection(series)
    
    def _quadratic_safe(self, series):
        """Enhanced quadratic interpolation"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 4:
            return self._linear_interpolation_safe(series)
        
        try:
            degree = min(2, len(values) - 1)
            coeffs = np.polyfit(indices, values, degree)
            
            all_indices = np.arange(len(series), dtype=np.float64)
            result = np.polyval(coeffs, all_indices)
            
            # More conservative extrapolation bounds
            value_range = np.ptp(values)
            if value_range > 0:
                center = np.median(values)
                max_deviation = 2.5 * value_range  # Tighter bounds
                result = np.clip(result, center - max_deviation, center + max_deviation)
            
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
            
        except Exception:
            return self._linear_interpolation_safe(series)
    
    def _periodic_safe(self, series):
        """Enhanced periodic interpolation with better smoothing"""
        arr = self._safe_array_conversion(series)
        indices, values, valid_mask = self._extract_valid_data(arr)
        
        if len(values) < 12:
            return self._smart_spline_selection(series)
        
        try:
            # Remove trend more carefully
            trend_coeffs = np.polyfit(indices, values, 1)
            trend = np.polyval(trend_coeffs, indices)
            detrended = values - trend
            
            # Apply adaptive smoothing
            if len(detrended) >= 9:
                window_length = min(len(detrended) // 3, 21)
                window_length = max(5, window_length)
                if window_length % 2 == 0:
                    window_length += 1
                
                try:
                    # Use lower polynomial order for noisy data
                    poly_order = min(3, window_length - 1)
                    smoothed = savgol_filter(detrended, window_length, poly_order)
                except Exception:
                    # Fallback to median filter
                    smoothed = median_filter(detrended, size=min(5, len(detrended)))
            else:
                smoothed = detrended
            
            # Use PCHIP for smoother interpolation
            f = interpolate.PchipInterpolator(indices, smoothed, extrapolate=True)
            
            all_indices = np.arange(len(series), dtype=np.float64)
            full_trend = np.polyval(trend_coeffs, all_indices)
            periodic_part = f(all_indices)
            
            result = full_trend + periodic_part
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return result
            
        except Exception:
            return self._smart_spline_selection(series)
    
    def impute_series(self, series):
        """Enhanced imputation with better method selection"""
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
                return self._linear_interpolation_safe(series).tolist()
            
            # Enhanced pattern detection
            pattern, confidence = self._enhanced_pattern_detection(indices, values)
            
            # Apply method with stricter thresholds
            if pattern == 'exponential' and confidence > 0.85:
                result = self._exponential_safe(series)
            elif pattern == 'periodic' and confidence > 0.6:
                result = self._periodic_safe(series)
            elif pattern == 'quadratic' and confidence > 0.75:
                result = self._quadratic_safe(series)
            else:
                # Default to smart spline selection
                result = self._smart_spline_selection(series)
            
            # Final validation
            result = np.asarray(result, dtype=np.float64)
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return [float(x) for x in result]
            
        except Exception as e:
            logger.error(f"Imputation error: {str(e)}")
            # Safe fallback
            valid_vals = [x for x in series if x is not None]
            if valid_vals:
                mean_val = sum(valid_vals) / len(valid_vals)
                return [float(x) if x is not None else float(mean_val) for x in series]
            else:
                return [0.0] * len(series)

# Initialize enhanced imputer
imputer = EnhancedImputer()

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