# archivo: qaoa_ciclovias.py
import json
import networkx as nx
import numpy as np
import os # Importar el módulo os


from qiskit.primitives import Sampler, StatevectorSampler # Importar StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
# ---------- Clase para representar la data extraída de OSM ----------
class OSMGraph:
    def __init__(self, osm_json):
        """
        osm_json: dict con keys 'nodes' y 'ways' como el JSON de ejemplo.
        Construye un grafo networkx con atributos en las aristas.
        """
        self.G = nx.Graph()
        for n in osm_json.get("nodes", []):
            self.G.add_node(n["id"], lat=n.get("lat"), lon=n.get("lon"), tags=n.get("tags", {}))
        for w in osm_json.get("ways", []):
            # asumimos tramos entre nodes consecutivos en 'nodes'
            nodes = w["nodes"]
            for u, v in zip(nodes[:-1], nodes[1:]):
                eid = f"{w['id']}_{u}_{v}"
                tags = w.get("tags", {})
                length = w.get("length_m", None)
                width = w.get("width_m", None)
                # calculo costo base: longitud / ancho (ejemplo simple)
                cost = length if length is not None else 1.0
                # marcar no-uso: si access=no o bicycle=no
                no_use = (tags.get("access") == "no") or (tags.get("bicycle") == "no")
                self.G.add_edge(u, v, id=eid, tags=tags, length=length, width=width, cost=cost, no_use=no_use)

    def add_candidate_edges_for_isolated_nodes(self, max_distance=50000000, candidate_cost_multiplier=0):
        """
        Identifies isolated nodes and adds candidate edges to connect them to the graph.
        """
        isolated_nodes = [node for node, degree in self.G.degree() if degree == 0]
        connected_nodes = [node for node, degree in self.G.degree() if degree > 0]
        
        if not connected_nodes:
            print("No connected nodes in the graph to link isolated nodes to.")
            return

        # Cache connected node coordinates for efficiency
        connected_node_coords = {node: (self.G.nodes[node].get('lat'), self.G.nodes[node].get('lon'))
                                 for node in connected_nodes
                                 if self.G.nodes[node].get('lat') is not None and self.G.nodes[node].get('lon') is not None}
        
        if not connected_node_coords:
            print("No connected nodes with valid lat/lon data to link isolated nodes to.")
            return

        for iso_node in isolated_nodes:
            min_dist = float('inf')
            closest_connected_node = None
            
            # Get coordinates for the isolated node
            iso_lat = self.G.nodes[iso_node].get('lat')
            iso_lon = self.G.nodes[iso_node].get('lon')

            if iso_lat is None or iso_lon is None:
                print(f"Skipping isolated node {iso_node} due to missing lat/lon data.")
                continue

            for conn_node, (conn_lat, conn_lon) in connected_node_coords.items():
                # Calculate Euclidean distance as an approximation
                # For real-world applications, use haversine distance or osmnx for projected graphs
                dist = np.sqrt((iso_lat - conn_lat)**2 + (iso_lon - conn_lon)**2)

                if dist < min_dist and dist * 111000 < max_distance: # Approximate distance in meters
                    min_dist = dist
                    closest_connected_node = conn_node
            
            if closest_connected_node is not None:
                # Create a new candidate edge
                new_edge_id = f"candidate_{iso_node}_{closest_connected_node}"
                # Approximate length based on distance
                approx_length = min_dist * 111000 # Rough conversion from degree diff to meters
                candidate_cost = approx_length * candidate_cost_multiplier
                
                self.G.add_edge(iso_node, closest_connected_node, 
                                id=new_edge_id, 
                                tags={"highway": "path", "name": f"Candidate Path {iso_node}-{closest_connected_node}", "is_candidate": "yes", "access": "yes", "bicycle": "yes"},
                                length_m=approx_length,
                                width_m=2.0, # Default width for candidate path
                                cost=candidate_cost,
                                no_use=False)
                print(f"Added candidate edge between {iso_node} and {closest_connected_node}")
            else:
                print(f"Could not find a connected node within {max_distance}m for isolated node {iso_node}.")


# ---------- Clase para construir el QUBO / QuadraticProgram ----------
class QUBOBuilder:
    def __init__(self, graph: OSMGraph, lambda_no=1000.0, lambda_deg=5000.0, budget=None):
        """
        graph: OSMGraph
        lambda_no: penalizacion para aristas no-usables
        lambda_deg: penalizacion para grado >= 1 (se usa (1 - sum_incident)^2)
        budget: float or None
        """
        self.G = graph.G
        self.lambda_no = lambda_no
        self.lambda_deg = lambda_deg
        self.budget = budget
        
        self.existing_base_edges = [] # All original edges from data.json
        self.candidate_optimizable_edges = [] # Only dynamically added candidate edges
        self.constant_offset = 0.0

        # Separate edges into existing base and dynamically added candidate edges
        for u, v, data in self.G.edges(data=True):
            if data.get("tags", {}).get("is_candidate") == "yes":
                self.candidate_optimizable_edges.append(((u, v), data))
            else:
                self.existing_base_edges.append(((u, v), data))
                self.constant_offset += data.get("cost", 1.0) # Add cost of all existing edges to constant offset

        # map optimizable edge -> var name (only for candidate edges)
        self.edge_vars = { (u,v): f"x_{u}_{v}" for (u,v),_ in self.candidate_optimizable_edges }
        self.qp = QuadraticProgram("ciclovias")

    def build(self):
        # añadir variables binarias solo para candidate_optimizable_edges
        for (u,v), edata in self.candidate_optimizable_edges:
            name = self.edge_vars[(u,v)]
            self.qp.binary_var(name=name)

        # objetivo: suma de costos + penalizaciones laterales
        linear = {name: 0.0 for name in self.edge_vars.values()}
        quadratic = {}

        # Add cost of candidate_optimizable_edges
        for (u,v), edata in self.candidate_optimizable_edges:
            name = self.edge_vars[(u,v)]
            c = edata.get("cost", 1.0)
            linear[name] += c

            # penaliza no_use for candidate_optimizable_edges
            if edata.get("no_use", False):
                linear[name] += self.lambda_no

        # penalización por grado: para cada nodo v, (1 - (total_base_degree + sum_incident_optimizable))^2
        for node in self.G.nodes():
            total_base_degree_for_node = sum(1 for (u,v),_ in self.existing_base_edges if u == node or v == node)
            
            incident_optimizable_vars = []
            for (u,v), edata in self.candidate_optimizable_edges:
                if u == node or v == node:
                    incident_optimizable_vars.append(self.edge_vars[(u,v)])
            
            # Term: (1 - (total_base_degree + sum_incident_optimizable))^2
            # Let S = (total_base_degree + sum_incident_optimizable)
            # We want to minimize (1 - S)^2 = 1 - 2S + S^2
            # constant 1 is ignored
            # -2S = -2 * (total_base_degree + sum_incident_optimizable)
            # S^2 = (total_base_degree + sum_incident_optimizable)^2

            # Linear part from -2S
            for var in incident_optimizable_vars:
                linear[var] += -2.0 * self.lambda_deg

            # Quadratic part from S^2
            for i in range(len(incident_optimizable_vars)):
                for j in range(i, len(incident_optimizable_vars)):
                    vi = incident_optimizable_vars[i]
                    vj = incident_optimizable_vars[j]
                    if vi == vj:
                        # diagonal contributes to linear: + lambda_deg * x_i (since x_i^2 = x_i)
                        linear[vi] += self.lambda_deg
                    else:
                        pair = tuple(sorted([vi, vj]))
                        quadratic[pair] = quadratic.get(pair, 0.0) + self.lambda_deg
            
            # Adjust linear terms for total_base_degree influence on -2S
            for var in incident_optimizable_vars:
                linear[var] += -2.0 * self.lambda_deg * total_base_degree_for_node

            # Constant offset from S^2, and from 1 in (1-S)^2 
            # This is complex, will simplify by adding to overall constant_offset
            self.constant_offset += self.lambda_deg * (total_base_degree_for_node**2 - 2 * total_base_degree_for_node + 1) # Full (1 - total_base_degree_for_node)^2

        # Apply overall constant offset to the objective
        self.qp.minimize(linear=linear, quadratic=quadratic, constant=self.constant_offset)

        # Budget constraint opcional - only for candidate_optimizable_edges costs now
        if self.budget is not None:
            cost_lin = {}
            current_total_base_cost = 0.0
            for (u,v), edata in self.existing_base_edges:
                current_total_base_cost += edata.get("cost", 1.0)

            for (u,v), edata in self.candidate_optimizable_edges:
                name = self.edge_vars[(u,v)]
                cost_lin[name] = edata.get("cost", 1.0)
            
            # The budget constraint now considers only optimizable edges' costs relative to the remaining budget
            remaining_budget = self.budget - current_total_base_cost
            self.qp.linear_constraint(linear=cost_lin, sense="<=", rhs=remaining_budget, name="budget")

        # Forzar no_use = 0 (opcional, más fuerte que penalización) - only for candidate_optimizable_edges
        for (u,v), edata in self.candidate_optimizable_edges:
            if edata.get("no_use", False):
                name = self.edge_vars[(u,v)]
                self.qp.linear_constraint(linear={name: 1}, sense="==", rhs=0, name=f"no_use_{name}")

        return self.qp


# ---------- Clase que ejecuta QAOA con Qiskit ----------
class QAOASolver:
    def __init__(self, quadr_program: QuadraticProgram, p=1, shots=1024, seed=42):
        self.qp = quadr_program
        self.p = p
        self.shots = shots
        self.seed = seed

    def solve(self):
        # convertir problema a QUBO y luego a Ising Hamiltonian (esto lo maneja internamente MinimumEigenOptimizer)
        # conv = QuadraticProgramToQubo()
        # qubo = conv.convert(self.qp)
        # ising_op, offset = to_ising(qubo)

        # print(f"Tipo de self.qp antes de la conversión a QUBO: {type(self.qp)}") # Eliminar línea de diagnóstico

        # construir QAOA
        # backend = Aer.get_backend('aer_simulator') # Ya no es necesario importar Aer
        # qi = QuantumInstance(backend, shots=self.shots, seed_simulator=self.seed, seed_transpiler=self.seed)
        sampler = StatevectorSampler() # Usar StatevectorSampler en lugar de Sampler

        opt = COBYLA(maxiter=200)
        # qaoa = QAOA(optimizer=opt, reps=self.p, sampler=sampler) # Usar sampler en lugar de quantum_instance
        qaoa = QAOA(sampler=sampler, optimizer=opt, reps=self.p, initial_point=np.array([0.0, 0.0])) # Added initial_point
        optimizer = MinimumEigenOptimizer(qaoa)

        result = optimizer.solve(self.qp) # Pasar el QuadraticProgram directamente
        return result


# ---------- Ejemplo de uso ----------
if __name__ == "__main__":
    # carga JSON (simula lo que obtendrías de OSM)
    data_json_path = os.path.join(os.path.dirname(__file__), "data.json")
    print(f"Intentando abrir el archivo: {data_json_path}") # Añadir línea para imprimir la ruta
    try:
        with open(data_json_path, "r") as f:
            osm_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo data.json en la ruta: {data_json_path}")
        exit(1)

    osm_graph = OSMGraph(osm_json)
    print("\n--- Initial Graph from data.json ---")
    print("Nodes:", osm_graph.G.nodes(data=True))
    print("Edges:", osm_graph.G.edges(data=True))

    # Add candidate edges for isolated nodes
    osm_graph.add_candidate_edges_for_isolated_nodes(max_distance=500, candidate_cost_multiplier=1.5)
    
    print("\n--- Graph Edges After Candidate Edge Addition ---")
    for u, v, data in osm_graph.G.edges(data=True):
        print(f"Edge ({u}, {v}): {data}")

    builder = QUBOBuilder(osm_graph, lambda_no=1000.0, lambda_deg=5000.0, budget=None)
    qp = builder.build()

    solver = QAOASolver(qp, p=1, shots=1024)
    result = solver.solve()

    print("Resultado QAOA:")
    print(result)

    print("\n--- Analysis of Optimized Solution ---")
    total_optimized_cost = 0.0
    # Map solution_x (numpy array) back to optimizable edge variables
    var_name_to_index = {var.name: i for i, var in enumerate(builder.qp.variables)}

    selected_optimizable_edges = []
    for (u, v), data in builder.candidate_optimizable_edges:
        var_name = f"x_{u}_{v}"
        if var_name in var_name_to_index:
            var_index = var_name_to_index[var_name]
            if result.x[var_index] == 1.0:
                selected_optimizable_edges.append(((u, v), data))
                total_optimized_cost += data.get("cost", 1.0)
    
    print(f"Total Cost of Selected Optimized Bike Paths (excluding fixed): {total_optimized_cost:.2f}")
    print("Selected Optimized Bike Paths:")
    if selected_optimizable_edges:
        for (u,v), data in selected_optimizable_edges:
            print(f"  Edge ({u}, {v}) - Cost: {data.get("cost", 1.0):.2f} - Tags: {data.get("tags",{})}")
    else:
        print("  No optimizable bike paths were selected.")

    print("Fixed Existing Bike Paths:")
    if builder.existing_base_edges:
        for (u,v), data in builder.existing_base_edges:
            print(f"  Edge ({u}, {v}) - Cost: {data.get("cost", 1.0):.2f} - Tags: {data.get("tags",{})}")
    else:
        print("  No fixed existing bike paths.")

    print("\n--- Node Connectivity Status ---")
    final_selected_edges = []
    # Add all existing base edges
    for (u, v), data in builder.existing_base_edges:
        final_selected_edges.append((u, v))
    # Add selected optimizable edges
    for (u,v), data in selected_optimizable_edges:
        final_selected_edges.append((u,v))

    # Create a temporary graph to check connectivity
    connectivity_graph = nx.Graph()
    connectivity_graph.add_nodes_from(osm_graph.G.nodes())
    connectivity_graph.add_edges_from(final_selected_edges)

    node_connectivity_status = {}
    for node in osm_graph.G.nodes():
        if connectivity_graph.degree(node) > 0:
            node_connectivity_status[node] = "Connected"
        else:
            node_connectivity_status[node] = "NOT Connected"

    print("  ", node_connectivity_status)

    print("\n--- Edge Connectivity Status ---")
    for u, v, data in osm_graph.G.edges(data=True):
        edge_key = (u, v)
        status = "connected" if edge_key in final_selected_edges else "not connected"
        print(f"  Edge {u}_{v}: {status}")

    # result.x viene con la asignacion de variables (0/1)
    # puedes mapear de vuelta a aristas usando builder.edge_vars

    # Export the solution to a JSON file
    def export_solution_to_json(graph, solution_x, builder, output_filename="data_output.json"):
        output_data = {"nodes": [], "ways": []}

        # Add nodes
        for node_id, node_data in graph.nodes(data=True):
            output_data["nodes"].append({"id": node_id, **node_data})

        # Add fixed edges (existing bike paths)
        for (u, v), data in builder.existing_base_edges:
            output_data["ways"].append({
                "id": data.get("id", f"way_{u}_{v}"),
                "nodes": [u, v],
                "tags": data.get("tags", {}),
                "length_m": data.get("length_m"),
                "width_m": data.get("width_m")
            })

        # Map solution_x (numpy array) back to optimizable edge variables
        var_name_to_index = {var.name: i for i, var in enumerate(builder.qp.variables)}
        
        for (u, v), data in builder.candidate_optimizable_edges:
            var_name = f"x_{u}_{v}"
            if var_name in var_name_to_index:
                var_index = var_name_to_index[var_name]
                if solution_x[var_index] == 1.0:
                    output_data["ways"].append({
                        "id": data.get("id", f"way_{u}_{v}"),
                        "nodes": [u, v],
                        "tags": data.get("tags", {}),
                        "length_m": data.get("length_m"),
                        "width_m": data.get("width_m")
                    })

        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Solution exported to {output_path}")

    export_solution_to_json(osm_graph.G, result.x, builder)
