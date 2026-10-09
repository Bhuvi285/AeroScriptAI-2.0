"""
AeroScript 2.0 — Smart Mathematical Engine
===========================================
Safely evaluates mathematical expressions drawn in the air.
Uses Python's Abstract Syntax Tree (AST) parser — completely secure, no eval().

Supports:
- Arithmetic: +, -, *, /, %
- Powers: ^ or **
- Parentheses: ( )
- Multi-digit numbers, negative numbers, floats
- Equalities: e.g. "2 + 3 =" or "2 + 3" -> "2 + 3 = 5"
"""

import ast
import operator
import re
import logging

log = logging.getLogger("AeroScript.Math")

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class MathEngine:
    """
    Evaluates handwritten arithmetic expressions safely.
    """

    @classmethod
    def sanitize(cls, raw_expr: str) -> str:
        """
        Cleans and normalizes air-written math strings.
        e.g., '2 x 3 =' -> '2 * 3'
        """
        expr = raw_expr.strip()
        # Remove trailing '=' or '?'
        expr = re.sub(r"[=\?]+$", "", expr).strip()
        
        # Replace multiplication representations
        # 'x' or 'X' between digits or parentheses -> '*'
        expr = re.sub(r"(?<=[0-9\)])\s*[xX]\s*(?=[0-9\(])", " * ", expr)
        # standalone 'x' or 'X'
        expr = expr.replace('x', '*').replace('X', '*')
        
        # Replace power notation
        expr = expr.replace('^', '**')
        
        # Replace division symbols if any
        expr = expr.replace('÷', '/').replace(':', '/')
        
        return expr

    @classmethod
    def _eval_node(cls, node):
        if isinstance(node, ast.Expression):
            return cls._eval_node(node.body)
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            if op_type == ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero")
            return SAFE_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = cls._eval_node(node.operand)
            return SAFE_OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    @classmethod
    def evaluate(cls, raw_text: str) -> dict:
        """
        Parses and evaluates an expression string.
        Returns:
            {
                "valid": bool,
                "clean_expr": str,
                "result": int | float | None,
                "formatted": str,   # e.g. "2 + 3 = 5"
                "error": str | None
            }
        """
        clean_expr = cls.sanitize(raw_text)
        if not clean_expr:
            return {
                "valid": False,
                "clean_expr": "",
                "result": None,
                "formatted": "",
                "error": "Empty expression"
            }
            
        try:
            tree = ast.parse(clean_expr, mode='eval')
            val = cls._eval_node(tree)
            
            # Format nicely
            if isinstance(val, float):
                if val.is_integer():
                    res = int(val)
                else:
                    res = round(val, 4)
            else:
                res = val
                
            formatted = f"{clean_expr.replace('**', '^')} = {res}"
            return {
                "valid": True,
                "clean_expr": clean_expr,
                "result": res,
                "formatted": formatted,
                "error": None
            }
        except ZeroDivisionError:
            return {
                "valid": False,
                "clean_expr": clean_expr,
                "result": None,
                "formatted": f"{clean_expr} = Inf",
                "error": "Division by zero"
            }
        except Exception as ex:
            log.debug(f"Math parse error for '{raw_text}': {ex}")
            return {
                "valid": False,
                "clean_expr": clean_expr,
                "result": None,
                "formatted": raw_text,
                "error": str(ex)
            }
