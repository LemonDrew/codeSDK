from flask import Flask, request, jsonify
import re
import math
from typing import Dict

from routes import app

class LaTeXFormulaEvaluator:
    def __init__(self):
        # Mapping of LaTeX functions to Python equivalents
        self.function_map = {
            'max': 'max',
            'min': 'min',
            'log': 'math.log',
            'ln': 'math.log',
            'exp': 'math.exp',
            'sqrt': 'math.sqrt',
            'sin': 'math.sin',
            'cos': 'math.cos',
            'tan': 'math.tan'
        }
    
    def clean_formula(self, formula: str) -> str:
        """Clean and normalize the LaTeX formula"""
        # Remove $$ delimiters
        formula = re.sub(r'\$+', '', formula)
        
        # Remove assignment part (everything before =)
        if '=' in formula:
            formula = formula.split('=', 1)[1].strip()
        
        return formula.strip()
    
    def replace_variables(self, formula: str, variables: Dict[str, float]) -> str:
        """Replace LaTeX text variables with their values"""
        # Handle \text{VariableName} format
        for var_name, value in variables.items():
            # Replace \text{VarName} with value
            pattern = r'\\text\{' + re.escape(var_name) + r'\}'
            formula = re.sub(pattern, str(value), formula)
            
            # Also handle direct variable names (for cases like E_R_m, beta_i, etc.)
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(var_name) + r'\b'
            formula = re.sub(pattern, str(value), formula)
        
        return formula
    
    def convert_latex_to_python(self, formula: str) -> str:
        """Convert LaTeX mathematical notation to Python"""
        # Replace LaTeX multiplication symbols
        formula = re.sub(r'\\times', '*', formula)
        formula = re.sub(r'\\cdot', '*', formula)
        
        # Replace LaTeX fractions \frac{a}{b} with (a)/(b)
        def replace_frac(match):
            numerator = match.group(1)
            denominator = match.group(2)
            return f'({numerator})/({denominator})'
        
        # Handle nested braces in fractions
        frac_pattern = r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        while re.search(frac_pattern, formula):
            formula = re.sub(frac_pattern, replace_frac, formula)
        
        # Replace LaTeX functions
        for latex_func, python_func in self.function_map.items():
            formula = re.sub(f'\\\\{latex_func}', python_func, formula)
        
        # Handle max and min functions
        formula = re.sub(r'\\max', 'max', formula)
        formula = re.sub(r'\\min', 'min', formula)
        
        # Handle exponentials e^x -> math.exp(x)
        def replace_exp(match):
            exponent = match.group(1)
            if exponent.startswith('{') and exponent.endswith('}'):
                exponent = exponent[1:-1]
            return f'math.exp({exponent})'
        
        formula = re.sub(r'e\^(\{[^}]+\}|\w+)', replace_exp, formula)
        
        # Handle other exponentials a^b -> pow(a, b)
        def replace_power(match):
            base = match.group(1)
            exponent = match.group(2)
            if exponent.startswith('{') and exponent.endswith('}'):
                exponent = exponent[1:-1]
            return f'pow({base}, {exponent})'
        
        formula = re.sub(r'([a-zA-Z0-9_.]+)\^(\{[^}]+\}|\w+)', replace_power, formula)
        
        # Handle summations (basic case)
        # \sum_{i=1}^{n} expression -> sum([expression for i in range(1, n+1)])
        # This is a simplified version - might need more complex handling for real cases
        
        # Clean up any remaining LaTeX commands
        formula = re.sub(r'\\[a-zA-Z]+', '', formula)
        
        # Replace remaining braces with parentheses
        formula = formula.replace('{', '(').replace('}', ')')
        
        return formula
    
    def evaluate_formula(self, formula: str, variables: Dict[str, float]) -> float:
        """Evaluate the LaTeX formula with given variables"""
        try:
            # Clean the formula
            cleaned_formula = self.clean_formula(formula)
            
            # Replace variables with their values
            formula_with_values = self.replace_variables(cleaned_formula, variables)
            
            # Convert LaTeX to Python
            python_expression = self.convert_latex_to_python(formula_with_values)
            
            # Evaluate the expression
            # Create a safe namespace for evaluation
            safe_dict = {
                'math': math,
                'max': max,
                'min': min,
                'pow': pow,
                'abs': abs,
                'round': round,
                '__builtins__': {}
            }
            
            result = eval(python_expression, safe_dict)
            return float(result)
            
        except Exception as e:
            raise ValueError(f"Error evaluating formula: {str(e)}")

@app.route('/trading-formula', methods=['POST'])
def trading():
    """
    Endpoint to evaluate LaTeX formulas for financial analysis
    
    Expected input: JSON array of test cases
    Expected output: JSON array of results
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({"error": "Expected JSON array"}), 400
        
        evaluator = LaTeXFormulaEvaluator()
        results = []
        
        for test_case in data:
            try:
                # Extract required fields
                name = test_case.get('name')
                formula = test_case.get('formula')
                variables = test_case.get('variables', {})
                test_type = test_case.get('type')
                
                if not formula or not isinstance(variables, dict):
                    results.append({"error": f"Invalid test case format for {name}"})
                    continue
                
                if test_type == 'compute':
                    # Evaluate the formula
                    result = evaluator.evaluate_formula(formula, variables)
                    # Round to 4 decimal places
                    rounded_result = round(result, 4)
                    results.append({"result": rounded_result})
                else:
                    results.append({"error": f"Unknown test type: {test_type}"})
                    
            except Exception as e:
                results.append({"error": f"Error processing test case: {str(e)}"})
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)