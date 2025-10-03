"""Health check endpoint"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        Simple status object
    """
    return {"status": "ok"}
