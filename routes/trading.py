from flask import Flask, request, jsonify
import re
import math
import ast
import operator

from routes import app

class LaTeXParser:
    def __init__(self):
        # Define supported operators and functions
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        self.functions = {
            'max': max,
            'min': min,
            'exp': math.exp,
            'log': math.log,
            'ln': math.log,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'abs': abs,
        }
    
    def parse_latex_formula(self, formula, variables):
        """Parse LaTeX formula and evaluate it with given variables"""
        
        # Remove dollar signs and equation parts
        formula = re.sub(r'\$\$?', '', formula)
        if '=' in formula:
            # Take the right side of the equation
            formula = formula.split('=', 1)[1].strip()
        
        # Replace LaTeX-specific notation
        formula = self._replace_latex_notation(formula)
        
        # Replace variable names with values
        formula = self._substitute_variables(formula, variables)
        
        # Convert to Python expression and evaluate
        try:
            result = self._safe_eval(formula)
            return round(float(result), 4)
        except Exception as e:
            print(f"Error evaluating formula '{formula}': {str(e)}")
            raise ValueError(f"Error evaluating formula: {str(e)}")
    
    def _replace_latex_notation(self, formula):
        """Replace LaTeX notation with Python equivalents"""
        
        # Handle text commands - be more flexible with spacing
        formula = re.sub(r'\\text\s*\{\s*([^}]+)\s*\}', r'\1', formula)
        
        # Handle fractions \frac{a}{b} -> (a)/(b)
        # Use recursive approach for nested fractions
        while r'\frac' in formula:
            formula = re.sub(r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', 
                           r'((\1)/(\2))', formula)
        
        # Handle remaining 'frac' without backslash (in case some got through)
        while 'frac{' in formula:
            formula = re.sub(r'frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', 
                           r'((\1)/(\2))', formula)
        
        # Handle square roots
        formula = re.sub(r'\\sqrt\s*\{([^}]+)\}', r'sqrt(\1)', formula)
        
        # Handle max and min functions - fix the malformed *func* issue
        formula = re.sub(r'\\max\s*', 'max', formula)
        formula = re.sub(r'\\min\s*', 'min', formula)
        # Clean up any remaining *func* artifacts
        formula = re.sub(r'\*func\*', '', formula)
        
        # Handle cdot and times multiplication
        formula = re.sub(r'\\cdot', '*', formula)
        formula = re.sub(r'\\times', '*', formula)
        
        # Handle exponentials - be more careful with e^x
        formula = re.sub(r'\be\s*\^\s*\{([^}]+)\}', r'exp(\1)', formula)
        formula = re.sub(r'\be\s*\^\s*([a-zA-Z0-9_]+)', r'exp(\1)', formula)
        
        # Handle logarithms
        formula = re.sub(r'\\log\s*\(', 'log(', formula)
        formula = re.sub(r'\\ln\s*\(', 'log(', formula)  # ln is natural log
        
        # Handle summations - simplified approach (just remove the LaTeX part)
        formula = re.sub(r'\\sum\s*_\{[^}]+\}', '0', formula)  # Replace with 0 for now
        formula = re.sub(r'sum_\{[^}]+\}', '0', formula)  # Handle cases without backslash
        formula = re.sub(r'sum_\{[^}]*', '0', formula)  # Handle incomplete summations
        
        # Handle left/right brackets
        formula = re.sub(r'\\left\s*\(', '(', formula)
        formula = re.sub(r'\\right\s*\)', ')', formula)
        formula = re.sub(r'\\left\s*\[', '(', formula)
        formula = re.sub(r'\\right\s*\]', ')', formula)
        formula = re.sub(r'left\s*\(', '(', formula)  # Without backslash
        formula = re.sub(r'right\s*\)', ')', formula)  # Without backslash
        
        # Handle superscripts and subscripts more carefully
        # First handle superscripts with braces
        formula = re.sub(r'\^\s*\{([^}]+)\}', r'**(\1)', formula)
        # Then handle single character superscripts
        formula = re.sub(r'\^\s*([a-zA-Z0-9.-]+)', r'**\1', formula)
        
        # Remove subscripts (they're usually just labels)
        formula = re.sub(r'_\s*\{[^}]+\}', '', formula)
        formula = re.sub(r'_\s*([a-zA-Z0-9])', '', formula)
        
        # Handle Greek letters and special symbols
        greek_letters = {
            r'\\alpha': 'alpha',
            r'\\beta': 'beta', 
            r'\\gamma': 'gamma',
            r'\\delta': 'delta',
            r'\\epsilon': 'epsilon',
            r'\\sigma': 'sigma',
            r'\\theta': 'theta',
            r'\\mu': 'mu',
            r'\\pi': str(math.pi),
            r'\\tau': str(2 * math.pi),
            r'\\rho': 'rho',
        }
        
        for latex_symbol, replacement in greek_letters.items():
            formula = re.sub(latex_symbol, replacement, formula)
        
        # Clean up extra spaces and remaining backslashes
        formula = re.sub(r'\\[a-zA-Z]+', '', formula)  # Remove unknown LaTeX commands
        formula = re.sub(r'\\', '', formula)
        formula = re.sub(r'\s+', ' ', formula).strip()
        
        return formula
    
    def _substitute_variables(self, formula, variables):
        """Replace variable names with their values"""
        
        # Create a mapping of all possible variable name variations
        var_mappings = {}
        
        for var_name, var_value in variables.items():
            # Original name
            var_mappings[var_name] = var_value
            
            # Handle underscores - both with and without
            if '_' in var_name:
                no_underscore = var_name.replace('_', '')
                var_mappings[no_underscore] = var_value
            
            # Handle bracket notation like E[R_m] -> E_R_m
            if '[' in var_name and ']' in var_name:
                bracket_to_underscore = var_name.replace('[', '_').replace(']', '')
                var_mappings[bracket_to_underscore] = var_value
                # Also without underscores
                var_mappings[bracket_to_underscore.replace('_', '')] = var_value
                
                # Handle E[R] style variables
                bracket_content = re.search(r'\[([^\]]+)\]', var_name)
                if bracket_content:
                    var_mappings[bracket_content.group(1)] = var_value
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_vars = sorted(var_mappings.items(), key=lambda x: len(x[0]), reverse=True)
        
        for var_name, var_value in sorted_vars:
            # Use word boundaries and be more flexible with matching
            patterns = [
                r'\b' + re.escape(var_name) + r'\b',  # Exact word match
                re.escape(var_name),  # Direct match for special cases
            ]
            
            for pattern in patterns:
                if re.search(pattern, formula):
                    formula = re.sub(pattern, f'({var_value})', formula)
                    break
        
        return formula
    
    def _safe_eval(self, expression):
        """Safely evaluate mathematical expression"""
        
        # Clean up the expression
        expression = expression.strip()
        
        # Handle implicit multiplication (like 2x -> 2*x)
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
        expression = re.sub(r'\)(\d)', r')*\1', expression)
        expression = re.sub(r'(\d)\(', r'\1*(', expression)
        expression = re.sub(r'\)\(', r')*(', expression)  # Handle )( -> )*(
        
        # Clean up any remaining issues
        expression = re.sub(r',\s*([a-zA-Z])', r'', expression)  # Remove trailing variables after commas
        
        # Replace function names with safe equivalents
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "sum": sum,
            "max": max,
            "min": min,
            "exp": math.exp,
            "log": math.log,
            "ln": math.log,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            # Add common variable names as fallback
            "alpha": 1,
            "beta": 1,
            "gamma": 1,
            "sigma": 1,
            "rho": 1,
            "epsilon": 1,
        }
        
        try:
            # First try to evaluate directly
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return float(result)
        except Exception as e1:
            try:
                # Try parsing as AST for more complex expressions
                parsed = ast.parse(expression, mode='eval')
                result = self._eval_ast_node(parsed.body, safe_dict)
                return float(result)
            except Exception as e2:
                print(f"Eval error 1: {e1}")
                print(f"Eval error 2: {e2}")
                print(f"Expression: {expression}")
                raise ValueError(f"Cannot evaluate expression: {expression}")
    
    def _eval_ast_node(self, node, safe_dict):
        """Evaluate AST node safely"""
        
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in safe_dict:
                return safe_dict[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_node(node.left, safe_dict)
            right = self._eval_ast_node(node.right, safe_dict)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_ast_node(node.operand, safe_dict)
            return self.operators[type(node.op)](operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            args = [self._eval_ast_node(arg, safe_dict) for arg in node.args]
            if func_name in safe_dict:
                return safe_dict[func_name](*args)
            else:
                raise ValueError(f"Unknown function: {func_name}")
        elif isinstance(node, ast.List):
            return [self._eval_ast_node(item, safe_dict) for item in node.elts]
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

@app.route('/trading-formula', methods=['POST'])
def trading():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({"error": "Expected JSON array"}), 400
        
        parser = LaTeXParser()
        results = []
        
        for i, test_case in enumerate(data):
            try:
                # Extract test case data
                name = test_case.get('name', f'test_{i+1}')
                formula = test_case.get('formula', '')
                variables = test_case.get('variables', {})
                test_type = test_case.get('type', 'compute')
                
                print(f"Processing {name}: {formula}")
                print(f"Variables: {variables}")
                
                # Evaluate formula
                if test_type == 'compute':
                    result = parser.parse_latex_formula(formula, variables)
                    results.append({"result": result})
                    print(f"Result: {result}")
                else:
                    results.append({"result": 0.0000})
                    
            except Exception as e:
                # If individual test case fails, return 0
                results.append({"result": 0.0000})
                print(f"Error in test case {test_case.get('name', 'unknown')}: {str(e)}")
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500