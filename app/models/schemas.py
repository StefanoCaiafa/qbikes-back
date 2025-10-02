"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field
from typing import List

class POI(BaseModel):
    """Point of Interest model"""
    lat: float = Field(..., description="Latitud del punto de interés")
    lon: float = Field(..., description="Longitud del punto de interés")
    weight: float = Field(..., description="Peso/importancia del POI", ge=0)

class OptimizationRequest(BaseModel):
    """Request model for bikeway optimization"""
    city: str = Field(
        ..., 
        description="Nombre de la ciudad o bounding box", 
        example="Piedmont, California, USA"
    )
    poi: List[POI] = Field(..., description="Lista de puntos de interés")
    budget_km: float = Field(
        ..., 
        description="Presupuesto máximo en km para nuevas ciclovías", 
        gt=0
    )
    lambda_cost: float = Field(
        default=1.0, 
        description="Peso para penalizar distancia/costo en la optimización",
        ge=0
    )

class EdgeInfo(BaseModel):
    """Information about a bikeway edge"""
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    length_m: float
    is_existing: bool = Field(
        default=False, 
        description="Si es ciclovía existente"
    )

class OptimizationResponse(BaseModel):
    """Response model for bikeway optimization"""
    selected_edges: List[EdgeInfo]
    total_length_km: float
    nodes_covered: int
    pois_covered: int
    existing_bikeways_km: float
    new_bikeways_km: float
