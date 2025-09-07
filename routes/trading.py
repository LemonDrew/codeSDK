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
        }
    
    def parse_latex_formula(self, formula, variables):
        """Parse LaTeX formula and evaluate it with given variables"""
        
        # Remove dollar signs and equation parts
        formula = re.sub(r'\$\$?', '', formula)
        if '=' in formula:
            # Take the right side of the equation
            formula = formula.split('=')[1].strip()
        
        # Replace LaTeX-specific notation
        formula = self._replace_latex_notation(formula)
        
        # Replace variable names with values
        formula = self._substitute_variables(formula, variables)
        
        # Convert to Python expression and evaluate
        try:
            result = self._safe_eval(formula)
            return round(result, 4)
        except Exception as e:
            raise ValueError(f"Error evaluating formula: {str(e)}")
    
    def _replace_latex_notation(self, formula):
        """Replace LaTeX notation with Python equivalents"""
        
        # Handle text commands
        formula = re.sub(r'\\text\{([^}]+)\}', r'\1', formula)
        
        # Handle fractions \frac{a}{b} -> (a)/(b)
        formula = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', formula)
        
        # Handle max and min functions
        formula = re.sub(r'\\max\s*\(', 'max(', formula)
        formula = re.sub(r'\\min\s*\(', 'min(', formula)
        
        # Handle cdot multiplication
        formula = re.sub(r'\\cdot', '*', formula)
        
        # Handle times multiplication
        formula = re.sub(r'\\times', '*', formula)
        
        # Handle exponentials
        formula = re.sub(r'e\^', 'exp(', formula)
        # Count braces to properly close exp function
        formula = self._handle_exponentials(formula)
        
        # Handle logarithms
        formula = re.sub(r'\\log\s*\(', 'log(', formula)
        formula = re.sub(r'\\ln\s*\(', 'ln(', formula)
        
        # Handle summations (basic case)
        formula = re.sub(r'\\sum', 'sum', formula)
        
        # Handle subscripts and superscripts (remove for basic evaluation)
        formula = re.sub(r'_\{[^}]+\}', '', formula)
        formula = re.sub(r'\^\{([^}]+)\}', r'**(\1)', formula)
        formula = re.sub(r'_([a-zA-Z0-9])', '', formula)
        formula = re.sub(r'\^([a-zA-Z0-9])', r'**\1', formula)
        
        # Handle Greek letters and special symbols
        formula = re.sub(r'\\alpha', 'alpha', formula)
        formula = re.sub(r'\\beta', 'beta', formula)
        formula = re.sub(r'\\sigma', 'sigma', formula)
        formula = re.sub(r'\\gamma', 'gamma', formula)
        formula = re.sub(r'\\delta', 'delta', formula)
        formula = re.sub(r'\\theta', 'theta', formula)
        formula = re.sub(r'\\mu', 'mu', formula)
        
        # Clean up extra spaces and backslashes
        formula = re.sub(r'\\', '', formula)
        formula = re.sub(r'\s+', ' ', formula).strip()
        
        return formula
    
    def _handle_exponentials(self, formula):
        """Handle exponential functions with proper parentheses"""
        result = ""
        i = 0
        while i < len(formula):
            if formula[i:i+4] == 'exp(':
                result += 'exp('
                i += 4
                paren_count = 1
                while i < len(formula) and paren_count > 0:
                    if formula[i] == '(':
                        paren_count += 1
                    elif formula[i] == ')':
                        paren_count -= 1
                    result += formula[i]
                    i += 1
            else:
                result += formula[i]
                i += 1
        return result
    
    def _substitute_variables(self, formula, variables):
        """Replace variable names with their values"""
        
        # Sort variables by length (longest first) to avoid partial replacements
        sorted_vars = sorted(variables.items(), key=lambda x: len(x[0]), reverse=True)
        
        for var_name, var_value in sorted_vars:
            # Handle different variable name formats
            var_patterns = [
                var_name,  # Direct name
                var_name.replace('_', ''),  # Remove underscores
            ]
            
            for pattern in var_patterns:
                # Use word boundaries to avoid partial replacements
                formula = re.sub(r'\b' + re.escape(pattern) + r'\b', str(var_value), formula)
        
        return formula
    
    def _safe_eval(self, expression):
        """Safely evaluate mathematical expression"""
        
        # Replace function names with safe equivalents
        for func_name, func in self.functions.items():
            expression = expression.replace(func_name, f'_func_{func_name}')
        
        # Create safe environment
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
        }
        
        # Add safe functions
        for func_name, func in self.functions.items():
            safe_dict[f'_func_{func_name}'] = func
        
        try:
            # Parse and evaluate the expression
            parsed = ast.parse(expression, mode='eval')
            result = self._eval_ast_node(parsed.body, safe_dict)
            return result
        except:
            # Fallback to direct eval for simple expressions
            try:
                return eval(expression, safe_dict)
            except:
                raise ValueError(f"Cannot evaluate expression: {expression}")
    
    def _eval_ast_node(self, node, safe_dict):
        """Evaluate AST node safely"""
        
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        elif isinstance(node, ast.Name):
            return safe_dict.get(node.id, 0)
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
        
        for test_case in data:
            try:
                # Extract test case data
                name = test_case.get('name', '')
                formula = test_case.get('formula', '')
                variables = test_case.get('variables', {})
                test_type = test_case.get('type', 'compute')
                
                # Evaluate formula
                if test_type == 'compute':
                    result = parser.parse_latex_formula(formula, variables)
                    results.append({"result": result})
                else:
                    results.append({"result": 0.0000})
                    
            except Exception as e:
                # If individual test case fails, return 0
                results.append({"result": 0.0000})
                print(f"Error in test case {test_case.get('name', 'unknown')}: {str(e)}")
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500