"""FastAPI application factory"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from .api.routes import health, bikeway


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="QBikes Bikeway Optimization API",
        description="API para optimizar redes de ciclovías usando datos de OpenStreetMap",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(bikeway.router)
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect to Swagger documentation"""
        return RedirectResponse(url="/docs")
    
    return app


# Create app instance
app = create_app()
