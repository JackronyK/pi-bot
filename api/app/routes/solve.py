# api/app/routes/solve.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TextSolveRequest(BaseModel):
    question: str

@router.post("/solve-text")
async def solve_text(req: TextSolveRequest):
    # For sprint 0 this is a stub; later it will enqueue a job.
    sample_answer = {
        "answer": "x = 2",
        "steps_html": "<p>Sample step: solve x+1=3</p>",
        "solver": "stub"
    }
    return sample_answer