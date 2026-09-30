"""Main API router.

Aggregates all sub-routers from various modules (authentication, etc.)
into a single router to be mounted on the FastAPI application.
"""

from fastapi import APIRouter

from erebots.api.routes import auth

router = APIRouter()

router.include_router(auth.router)
