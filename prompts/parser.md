### Task
Parse the user's math problem and return strict JSON only (no explanation, no markdown).

Required JSON schema:
{
  "problem_type": "solve" | "simplify" | "evaluate" | "other",
  "raw_question": "<original text>",
  "goal": "<short goal, e.g. 'solve'>",
  "equations": [ { "lhs": "<lhs string>", "rhs": "<rhs string>" }, ... ],
  "unknowns": ["x","y",...]
}

Rules:
- Output **only** JSON. Do not add text outside the JSON object.
- Use algebraic strings: prefer `2*x` or `2*x` not `2x`.
- If you detect an equation like `2x+3=11`, convert it to `2*x + 3` for clarity.
- If you cannot find any equation, set "equations": [].

Examples (few-shot):

Input: "Solve for x: 2*x + 3 = 11"
Output:
{
  "problem_type": "solve",
  "raw_question": "Solve for x: 2*x + 3 = 11",
  "goal": "solve",
  "equations": [{"lhs": "2*x + 3", "rhs": "11"}],
  "unknowns": ["x"]
}

Input: "Solve 3y-3=12"
Output:
{
  "problem_type": "solve",
  "raw_question": "Solve 3y-3=12",
  "goal": "solve",
  "equations": [{"lhs": "3*y - 3", "rhs": "12"}],
  "unknowns": ["y"]
}
