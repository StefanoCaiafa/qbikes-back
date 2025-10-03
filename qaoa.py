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


# ---------- Clase para construir el QUBO / QuadraticProgram ----------
class QUBOBuilder:
    def __init__(self, graph: OSMGraph, lambda_no=1000.0, lambda_deg=50.0, budget=None):
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
        # map edge -> var name
        self.edges = list(self.G.edges(data=True))
        self.edge_vars = { (u,v): f"x_{u}_{v}" for u,v,_ in self.edges }
        self.qp = QuadraticProgram("ciclovias")

    def build(self):
        # añadir variables binarias
        for (u,v,edata) in self.edges:
            name = self.edge_vars[(u,v)]
            self.qp.binary_var(name=name)

        # objetivo: suma de costos + penalizaciones laterales
        linear = {name: 0.0 for name in self.edge_vars.values()}
        quadratic = {}

        # costo base
        for (u,v,edata) in self.edges:
            name = self.edge_vars[(u,v)]
            c = edata.get("cost", 1.0)
            linear[name] += c

            # penaliza no_use
            if edata.get("no_use", False):
                linear[name] += self.lambda_no

        # penalización por grado: para cada nodo v, (1 - sum_incident(x))^2
        # expandimos: 1 - 2 sum x_i + sum_{i,j} x_i x_j
        for node in self.G.nodes():
            incident = []
            for nbr in self.G.neighbors(node):
                # edge key ordering might differ; ensure consistent mapping
                key = (node, nbr) if (node,nbr) in self.edge_vars else (nbr,node)
                incident.append(self.edge_vars[key])
            if not incident:
                # nodo aislado: penalización grande (no se podrá cumplir)
                continue
            # linear terms: -2 * lambda_deg * x_i
            for var in incident:
                linear[var] += -2.0 * self.lambda_deg
            # quadratic terms: + lambda_deg * x_i x_j for all pairs (including i==j but that's linear)
            for i in range(len(incident)):
                for j in range(i, len(incident)):
                    vi = incident[i]
                    vj = incident[j]
                    if vi == vj:
                        # diagonal contributes to linear: + lambda_deg * x_i    (since x_i^2 = x_i)
                        linear[vi] += self.lambda_deg
                    else:
                        pair = tuple(sorted([vi, vj]))
                        quadratic[pair] = quadratic.get(pair, 0.0) + self.lambda_deg

            # constant term (1) * lambda_deg is irrelevant for optimization so la omitimos

        # build objective in QuadraticProgram
        # set linear part
        for var,nameval in linear.items():
            # QuadraticProgram expects objective via set_linear / set_quadratic, we will assemble via .objective
            pass

        # qiskit-optimization: use .objective.set_linear / set_quadratic
        # self.qp.objective.sense = self.qp.objective.Sense.MINIMIZE # Sense is set by minimize method
        # # self.qp.objective.set_linear(linear)
        # # set quadratic
        # # for (a,b), coeff in quadratic.items():
        # #     self.qp.objective.set_quadratic(a, b, coeff)
        # self.qp.objective.set_coefficients(linear=linear, quadratic=quadratic)
        self.qp.minimize(linear=linear, quadratic=quadratic)

        # Budget constraint opcional
        if self.budget is not None:
            # sum_e cost_e x_e <= budget
            cost_lin = {}
            for (u,v,edata) in self.edges:
                name = self.edge_vars[(u,v)]
                cost_lin[name] = edata.get("cost", 1.0)
            self.qp.linear_constraint(linear=cost_lin, sense="<=", rhs=self.budget, name="budget")

        # Forzar no_use = 0 (opcional, más fuerte que penalización)
        # podemos agregar constraints var == 0 para edges con no_use True
        for (u,v,edata) in self.edges:
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
    builder = QUBOBuilder(osm_graph, lambda_no=1000.0, lambda_deg=50.0, budget=None)
    qp = builder.build()

    solver = QAOASolver(qp, p=1, shots=1024)
    result = solver.solve()

    print("Resultado QAOA:")
    print(result)
    # result.x viene con la asignacion de variables (0/1)
    # puedes mapear de vuelta a aristas usando builder.edge_vars

