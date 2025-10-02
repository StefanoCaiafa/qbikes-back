"""Bikeway optimization endpoints"""
from fastapi import APIRouter, HTTPException
from ...models.schemas import OptimizationRequest, OptimizationResponse
from ...services.osm_service import OSMService
from ...services.optimization_service import OptimizationService

router = APIRouter()


@router.post("/optimize_bikeway", response_model=OptimizationResponse, tags=["Optimization"])
async def optimize_bikeway(request: OptimizationRequest):
    """
    Optimize bikeway network for a given city.
    
    **Note:** Currently using simple heuristic optimization. 
    QAOA/QUBO quantum optimization is not yet implemented.
    
    This endpoint:
    1. Downloads street graph from OpenStreetMap
    2. Identifies existing bikeways
    3. Applies heuristic to select new bikeways
    4. Returns optimized solution
    
    Args:
        request: Optimization request with city, POIs, budget, and lambda_cost
    
    Returns:
        Optimization response with selected edges and metrics
    """
    try:
        # 1. Download graph
        print(f"[1/5] Downloading graph for {request.city}...")
        G = OSMService.download_graph(request.city)
        print(f"      ✓ Graph downloaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 2. Classify edges into existing and candidates
        print(f"[2/5] Classifying edges...")
        existing_edges, candidate_edges = OSMService.classify_edges(G)
        print(f"      ✓ Existing bikeways: {len(existing_edges)}")
        print(f"      ✓ Candidate edges: {len(candidate_edges)}")
        
        # 3. Calculate POI weights
        print(f"[3/5] Calculating POI weights...")
        pois_dict = [poi.dict() for poi in request.poi]
        poi_node_weights = OSMService.calculate_poi_weights(G, pois_dict, radius_m=100)
        print(f"      ✓ Nodes with nearby POIs: {len(poi_node_weights)}")
        
        # 4. Apply QAOA/QUBO optimization algorithm
        print(f"[4/5] Applying QAOA/QUBO optimization...")
        budget_m = request.budget_km * 1000
        
        selected_new_edges = OptimizationService.qaoa_optimization(
            candidate_edges, budget_m, poi_node_weights, request.lambda_cost
        )
        print(f"      ✓ New edges selected: {len(selected_new_edges)}")
        
        # 5. Build response
        print(f"[5/5] Building response...")
        result = OptimizationService.build_response(
            G,
            existing_edges,
            selected_new_edges,
            request.poi,
            poi_node_weights
        )
        print(f"      ✓ Optimization complete!")
        
        return OptimizationResponse(**result)
        
    except NotImplementedError as e:
        # Specific error for QAOA/QUBO not implemented
        raise HTTPException(
            status_code=501,  # Not Implemented
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in optimization: {str(e)}"
        )
