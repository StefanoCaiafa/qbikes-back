"""OpenStreetMap service for downloading and processing street graphs"""
import osmnx as ox
import networkx as nx
from typing import Tuple, List, Dict
from ..utils.geo_utils import haversine_distance


class OSMService:
    """Service for interacting with OpenStreetMap data"""
    
    @staticmethod
    def download_graph(city: str) -> nx.MultiDiGraph:
        """
        Download street graph from OpenStreetMap.
        
        Args:
            city: City name or bounding box
        
        Returns:
            NetworkX MultiDiGraph with street network
        """
        G = ox.graph_from_place(city, network_type='bike')
        # Convert to undirected graph using NetworkX
        G = G.to_undirected()
        return G
    
    @staticmethod
    def get_node_coords(G: nx.MultiDiGraph, node: int) -> Tuple[float, float]:
        """
        Get coordinates (lat, lon) of a node.
        
        Args:
            G: NetworkX graph
            node: Node ID
        
        Returns:
            Tuple of (latitude, longitude)
        """
        return G.nodes[node]['y'], G.nodes[node]['x']
    
    @staticmethod
    def is_existing_bikeway(edge_data: dict) -> bool:
        """
        Determine if an edge is an existing bikeway based on OSM attributes.
        
        Args:
            edge_data: Edge data dictionary from NetworkX
        
        Returns:
            True if the edge is an existing bikeway
        """
        # Check if highway is cycleway
        if edge_data.get('highway') == 'cycleway':
            return True
        
        # Check if it has cycleway attribute
        cycleway = edge_data.get('cycleway')
        if cycleway and cycleway != 'no':
            return True
        
        # Check other cycling-related attributes
        for key in ['cycleway:right', 'cycleway:left', 'cycleway:both']:
            if edge_data.get(key) and edge_data.get(key) != 'no':
                return True
        
        return False
    
    @staticmethod
    def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
        """
        Find the nearest node to a given point.
        
        Args:
            G: NetworkX graph
            lat: Latitude
            lon: Longitude
        
        Returns:
            Node ID of the nearest node
        """
        return ox.distance.nearest_nodes(G, lon, lat)
    
    @staticmethod
    def classify_edges(G: nx.MultiDiGraph) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Classify graph edges into existing bikeways and candidates.
        
        Args:
            G: NetworkX graph
        
        Returns:
            Tuple of (existing_edges, candidate_edges)
        """
        existing_edges = []
        candidate_edges = []
        
        for u, v, key, data in G.edges(keys=True, data=True):
            # Ensure length attribute exists
            if 'length' not in data:
                u_lat, u_lon = OSMService.get_node_coords(G, u)
                v_lat, v_lon = OSMService.get_node_coords(G, v)
                data['length'] = haversine_distance(u_lat, u_lon, v_lat, v_lon)
            
            if OSMService.is_existing_bikeway(data):
                data['fixed'] = 1
                existing_edges.append((u, v, key, data))
            else:
                data['fixed'] = 0
                candidate_edges.append((u, v, key, data))
        
        return existing_edges, candidate_edges
    
    @staticmethod
    def calculate_poi_weights(
        G: nx.MultiDiGraph, 
        pois: List[Dict], 
        radius_m: float = 100
    ) -> Dict[int, float]:
        """
        Calculate additional weights for nodes near POIs.
        
        Args:
            G: NetworkX graph
            pois: List of POIs with lat, lon, and weight
            radius_m: Radius in meters to consider a POI as covered
        
        Returns:
            Dictionary mapping node_id to weight
        """
        node_weights = {}
        
        for poi in pois:
            nearest_node = OSMService.find_nearest_node(G, poi['lat'], poi['lon'])
            node_lat, node_lon = OSMService.get_node_coords(G, nearest_node)
            
            # If node is within radius, assign weight
            distance = haversine_distance(poi['lat'], poi['lon'], node_lat, node_lon)
            if distance <= radius_m:
                if nearest_node not in node_weights:
                    node_weights[nearest_node] = 0
                node_weights[nearest_node] += poi['weight']
        
        return node_weights
