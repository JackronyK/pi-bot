import json
import sympy

def solve():
    steps = []
    
    # Define variable
    x = sympy.symbols('x')
    
    # Define equation sides based on input: 2*x + 3 = 11
    lhs = 2 * x + 3
    rhs = 11
    equation = sympy.Eq(lhs, rhs)
    
    steps.append(f"Start with the equation: $${sympy.latex(equation)}$$")
    
    # Step 1: Isolate the term with x (subtract 3 from both sides)
    # We conceptually move the constant term (3) to the RHS
    constant_term = 3
    new_rhs = rhs - constant_term
    term_with_x = lhs - constant_term
    
    steps.append(f"Subtract {constant_term} from both sides:")
    steps.append(f"$${sympy.latex(term_with_x)} = {sympy.latex(rhs)} - {constant_term}$$")
    steps.append(f"Simplify:")
    steps.append(f"$${sympy.latex(term_with_x)} = {sympy.latex(new_rhs)}$$")
    
    # Step 2: Solve for x (divide by coefficient 2)
    coefficient = 2
    final_lhs = term_with_x / coefficient
    final_rhs = new_rhs / coefficient
    
    steps.append(f"Divide both sides by {coefficient}:")
    steps.append(f"$$x = \\frac{{{sympy.latex(new_rhs)}}}{{{coefficient}}}$$")
    
    # Calculate final solution
    solution = sympy.solve(equation, x)
    
    # Format the final answer
    if solution:
        ans = solution[0]
        steps.append(f"Result: $$x = {sympy.latex(ans)}$$")
        answer_str = str(ans)
    else:
        steps.append("No solution found.")
        answer_str = "No solution"

    return {
        "answer": answer_str,
        "steps": steps,
        "structured": {
            "variable": "x",
            "value": str(solution[0]) if solution else None
        }
    }

if __name__ == "__main__":
    try:
        print(json.dumps(solve(), ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))