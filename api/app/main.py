# api/app/main.py

from fastapi import FastAPI
from .routes.health import router as health_router
from .routes.solve import router as solve_router

app = FastAPI(title="pibot-api", version="0.1")

app.include_router(health_router, prefix="/health")
app.include_router(solve_router, prefix="/api")

# Simple startup/ shutdown

