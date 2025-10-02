"""Optimization service for bikeway selection"""
from typing import List, Tuple, Dict, Set
import networkx as nx
from ..models.schemas import POI, EdgeInfo
from .osm_service import OSMService


class OptimizationService:
    """Service for optimizing bikeway network selection"""
    
    @staticmethod
    def qaoa_optimization(
        candidate_edges: List[Tuple],
        budget_m: float,
        poi_node_weights: Dict[int, float],
        lambda_cost: float
    ) -> List[Tuple]:
        """
        QAOA/QUBO optimization for bikeway selection.
        
        **NOT IMPLEMENTED YET**
        
        This method would implement quantum optimization using QAOA (Quantum Approximate 
        Optimization Algorithm) to solve the QUBO (Quadratic Unconstrained Binary Optimization)
        problem for optimal bikeway selection.
        
        Args:
            candidate_edges: List of candidate edges
            budget_m: Budget in meters
            poi_node_weights: Dictionary of node weights from POIs
            lambda_cost: Weight for distance/cost penalty
        
        Returns:
            List of selected edges
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "QAOA/QUBO optimization is not implemented yet. "
            "\\n\\nThis method needs to be implemented by the quantum optimization team. "
            "\\n\\nImplementation requirements:"
            "\\n- Use Qiskit or similar quantum computing framework"
            "\\n- Formulate bikeway selection as a QUBO problem"
            "\\n- Use QAOA to find optimal edge selection"
            "\\n- Maximize POI coverage within budget constraint"
            "\\n- Return List[Tuple] of selected edges in format: [(u, v, key, data), ...]"
        )
    
    @staticmethod
    def build_response(
        G: nx.MultiDiGraph,
        existing_edges: List[Tuple],
        selected_new_edges: List[Tuple],
        pois: List[POI],
        poi_node_weights: Dict[int, float]
    ) -> Dict:
        """
        Build optimization response with all metrics.
        
        Args:
            G: NetworkX graph
            existing_edges: List of existing bikeway edges
            selected_new_edges: List of newly selected edges
            pois: List of POIs
            poi_node_weights: Dictionary of node weights
        
        Returns:
            Dictionary with optimization results
        """
        result_edges = []
        total_new_length = 0.0
        total_existing_length = 0.0
        covered_nodes: Set[int] = set()
        
        # Add existing bikeways
        for u, v, key, data in existing_edges:
            u_lat, u_lon = OSMService.get_node_coords(G, u)
            v_lat, v_lon = OSMService.get_node_coords(G, v)
            
            result_edges.append(EdgeInfo(
                start_lat=u_lat,
                start_lon=u_lon,
                end_lat=v_lat,
                end_lon=v_lon,
                length_m=data['length'],
                is_existing=True
            ))
            
            total_existing_length += data['length']
            covered_nodes.add(u)
            covered_nodes.add(v)
        
        # Add newly selected bikeways
        for u, v, key, data in selected_new_edges:
            u_lat, u_lon = OSMService.get_node_coords(G, u)
            v_lat, v_lon = OSMService.get_node_coords(G, v)
            
            result_edges.append(EdgeInfo(
                start_lat=u_lat,
                start_lon=u_lon,
                end_lat=v_lat,
                end_lon=v_lon,
                length_m=data['length'],
                is_existing=False
            ))
            
            total_new_length += data['length']
            covered_nodes.add(u)
            covered_nodes.add(v)
        
        # Count covered POIs
        pois_covered = OptimizationService._count_covered_pois(
            G, pois, covered_nodes
        )
        
        return {
            "selected_edges": result_edges,
            "total_length_km": (total_existing_length + total_new_length) / 1000,
            "nodes_covered": len(covered_nodes),
            "pois_covered": pois_covered,
            "existing_bikeways_km": total_existing_length / 1000,
            "new_bikeways_km": total_new_length / 1000
        }
    
    @staticmethod
    def _count_covered_pois(
        G: nx.MultiDiGraph, 
        pois: List[POI], 
        covered_nodes: Set[int],
        radius_m: float = 100
    ) -> int:
        """
        Count how many POIs are covered by the bikeway network.
        
        Args:
            G: NetworkX graph
            pois: List of POIs
            covered_nodes: Set of nodes covered by bikeways
            radius_m: Radius in meters to consider a POI as covered
        
        Returns:
            Number of covered POIs
        """
        from ..utils.geo_utils import haversine_distance
        
        pois_covered = 0
        for poi in pois:
            nearest_node = OSMService.find_nearest_node(G, poi.lat, poi.lon)
            if nearest_node in covered_nodes:
                node_lat, node_lon = OSMService.get_node_coords(G, nearest_node)
                distance = haversine_distance(poi.lat, poi.lon, node_lat, node_lon)
                if distance <= radius_m:
                    pois_covered += 1
        
        return pois_covered
