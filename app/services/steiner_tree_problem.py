# qubo_bike_steiner.py
# QUBO para ciclovías (Prize-Collecting Steiner + mínimo 1 conexión entre componentes)
# Requiere: qiskit-terra, qiskit-algorithms, qiskit-aer, qiskit-optimization (según tu instalación)

from __future__ import annotations
import math
from collections import defaultdict

try:
    from data import NODES, EDGES, TERMINALS, CANDIDATE_STEINER, POI_PRIZES
except Exception:
    # Ejemplo mínimo si no existe data.py
    NODES = {
        1: {'lat': 0.0, 'lon': 0.0, 'kind': 'street',   'component': 1},
        2: {'lat': 0.0, 'lon': 1.0, 'kind': 'street',   'component': 1},
        3: {'lat': 0.0, 'lon': 2.0, 'kind': 'terminal', 'component': 1},
        4: {'lat': 1.0, 'lon': 0.5, 'kind': 'street',   'component': 2},
        5: {'lat': 1.0, 'lon': 1.5, 'kind': 'street',   'component': 2},
    }
    EDGES = [
        (1, 2, {'length': 0.8, 'name': 'Calle A', 'inter_component': False}),
        (2, 3, {'length': 0.5, 'name': 'Calle B', 'inter_component': False}),
        (4, 5, {'length': 0.6, 'name': 'Calle C', 'inter_component': False}),
        (2, 4, {'length': 1.2, 'name': 'Av. Conectora', 'inter_component': True}),
    ]
    TERMINALS = [3]
    CANDIDATE_STEINER = [1,2,4,5]
    POI_PRIZES = {}

# ------------------------------------------
# Qiskit imports (versiones modernas)
# ------------------------------------------
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator, Sampler
from qiskit_aer.primitives import Estimator as AerEstimator
import random

# Set random seed for reproducibility
random.seed(42)

def build_qubo(NODES, EDGES, TERMINALS, CANDIDATE_STEINER, POI_PRIZES,
               alpha=1.0,     # peso de premios a POIs opcionales
               lam1=10.0,     # edge => endpoints
               lam2=10.0,     # terminal must be selected
               lam3=5.0,      # terminal has at least one incident edge
               lam4=1.0,      # |E| == |V|-1 (aprox anti-ciclos)
               lam5=8.0       # al menos un puente entre componentes
               ) -> QuadraticProgram:

    qp = QuadraticProgram(name="BikeSteinerQUBO")

    # Mapas de índices para variables
    edge_ids = []
    for idx, (u,v,attr) in enumerate(EDGES):
        # normaliza (u<v)
        i, j = (u, v) if u < v else (v, u)
        edge_ids.append((i, j, attr))
    edge_var = {}  # (i,j) -> name
    for (i,j,attr) in edge_ids:
        name = f"x_{i}_{j}"
        qp.binary_var(name)
        edge_var[(i,j)] = name

    node_var = {}  # v -> name
    for v in NODES.keys():
        name = f"z_{v}"
        qp.binary_var(name)
        node_var[v] = name

    # --- Objetivo: costos de aristas y premios por POIs opcionales
    lin = defaultdict(float)
    quad = defaultdict(float)

    # Costos de arista
    for (i,j,attr) in edge_ids:
        w = float(attr.get('length', 1.0))
        lin[edge_var[(i,j)]] += w

    # Premios de POIs opcionales
    for v, prize in POI_PRIZES.items():
        if v in node_var:
            lin[node_var[v]] -= alpha * float(prize)

    # --- Penalización 1: arista implica extremos
    for (i,j,attr) in edge_ids:
        x = edge_var[(i,j)]
        zi, zj = node_var[i], node_var[j]
        # x*(1 - z_i) + x*(1 - z_j) = 2x - x z_i - x z_j
        lin[x] += lam1 * 2.0
        quad[(x, zi)] -= lam1
        quad[(x, zj)] -= lam1

    # --- Penalización 2: terminales deben seleccionarse (1 - z_t)^2
    for t in TERMINALS:
        zt = node_var[t]
        # (1 - z)^2 = 1 - 2z + z^2 ; como z es binaria, z^2=z, queda 1 - z
        # En QUBO puro se suele incluir constante, pero podemos omitirla para no desplazar.
        lin[zt] += lam2 * (-1.0)   # - lam2*z_t
        # (la constante lam2*1 la omitimos: solo desplaza)

    # --- Penalización 3: cada terminal con ≥1 arista incidente
    # (1 - sum x_e)^2 = 1 - 2 sum x + sum_{e,f} x_e x_f
    incident = defaultdict(list)
    for (i,j,attr) in edge_ids:
        incident[i].append(edge_var[(i,j)])
        incident[j].append(edge_var[(i,j)])

    for t in TERMINALS:
        inc_vars = incident[t]
        # -2 * sum x_e
        for xv in inc_vars:
            lin[xv] += lam3 * (-2.0)
        # + sum_{e} x_e^2  +  2 * sum_{e<f} x_e x_f
        # Como x^2 = x para binarias, el término x^2 lo podemos sumar a lineal
        for a in range(len(inc_vars)):
            lin[inc_vars[a]] += lam3 * 1.0
            for b in range(a+1, len(inc_vars)):
                quad[(inc_vars[a], inc_vars[b])] += lam3 * 2.0
        # constante +lam3*1 omitida

    # --- Penalización 4: |E| == |V|-1  => (sum x - sum z + 1)^2
    # Expande: (A - B + 1)^2 = A^2 + B^2 + 1 + cross - 2A B + 2A - 2B
    # Con binarias: A^2 -> suma x, B^2 -> suma z (porque x^2=x, z^2=z)
    # Coeficientes lineales:
    for (i,j,attr) in edge_ids:
        lin[edge_var[(i,j)]] += lam4 * (1.0 + 2.0)  # de A^2 (+1) y +2A
    for v in NODES.keys():
        lin[node_var[v]] += lam4 * (1.0 - 2.0)      # de B^2 (+1) y -2B

    # Cruzados: -2 A B  y (entre edges entre sí y nodes entre sí no hay por esta forma)
    # A= sum x ; B= sum z  => para cada par (x, z): coef -2
    for (i,j,attr) in edge_ids:
        x = edge_var[(i,j)]
        for v in NODES.keys():
            z = node_var[v]
            quad[(x, z)] += lam4 * (-2.0)
    # Constante lam4*(+1) omitida

    # --- Penalización 5: al menos un puente inter-componente
    cross_edges = [edge_var[(min(u,v), max(u,v))] for (u,v,attr) in edge_ids if bool(attr.get('inter_component', False))]

    if cross_edges:
        # (1 - sum x_cross)^2 = 1 - 2 sum x + sum x_i^2 + 2 sum_{i<j} x_i x_j
        for xv in cross_edges:
            lin[xv] += lam5 * (-2.0)   # -2 * sum x
            lin[xv] += lam5 * (1.0)    # + sum x_i (por x_i^2)
        for a in range(len(cross_edges)):
            for b in range(a+1, len(cross_edges)):
                quad[(cross_edges[a], cross_edges[b])] += lam5 * 2.0
        # constante lam5*1 omitida

    # Construir objetivo
    qp.minimize(linear=dict(lin), quadratic=dict(quad))
    return qp


def solve_qubo(qp: QuadraticProgram, shots: int = 2048, reps: int = 2):
    # Intenta usar Aer; si no, usa Estimator por defecto
    try:
        estimator = AerEstimator()
    except Exception:
        estimator = Estimator()
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=100), reps=reps)
    meo = MinimumEigenOptimizer(qaoa)
    try:
        result = meo.solve(qp)
        return result
    except Exception as e:
        # Fallback clásico si QAOA falla en tu entorno
        from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer as MEO
        try:
            # Si tienes CPLEX instalado…
            opt = CplexOptimizer()
            return opt.solve(qp)
        except Exception:
            # Último recurso: convertir a Ising y evaluar best sample con NumPy
            from qiskit_optimization.converters import QuadraticProgramToQubo
            from qiskit_algorithms import NumPyMinimumEigensolver
            conv = QuadraticProgramToQubo()
            qubo = conv.convert(qp)
            me = NumPyMinimumEigensolver()
            meo2 = MinimumEigenOptimizer(me)
            return meo2.solve(qubo)


def decode_solution(result, NODES, EDGES):
    x_take = []
    z_take = []
    for var, val in result.variables_dict.items():
        if val < 0.5:  # binario
            continue
        if var.startswith("x_"):
            _, i, j = var.split("_")
            i, j = int(i), int(j)
            x_take.append((i, j))
        elif var.startswith("z_"):
            _, v = var.split("_")
            z_take.append(int(v))

    # Aristas con metadata
    meta = {}
    for (u,v,attr) in EDGES:
        i,j = (u,v) if u<v else (v,u)
        meta[(i,j)] = attr

    chosen_edges = [(i, j, meta.get((min(i,j), max(i,j)), {})) for (i,j) in x_take]
    chosen_nodes = [(v, NODES[v]) for v in z_take]

    return chosen_nodes, chosen_edges


def main():
    qp = build_qubo(
        NODES=NODES,
        EDGES=EDGES,
        TERMINALS=TERMINALS,
        CANDIDATE_STEINER=CANDIDATE_STEINER,
        POI_PRIZES=POI_PRIZES,
        alpha=1.0, lam1=10.0, lam2=10.0, lam3=5.0, lam4=1.0, lam5=8.0
    )

    print(f"QUBO vars: {qp.get_num_vars()} (edges + nodes)")
    result = solve_qubo(qp, reps=2)
    print("Objective value:", result.fval)

    chosen_nodes, chosen_edges = decode_solution(result, NODES, EDGES)

    print("\n=== Nodos seleccionados (z_v=1) ===")
    for v, info in sorted(chosen_nodes):
        print(f"v={v} kind={info.get('kind')} comp={info.get('component')} lat={info.get('lat')} lon={info.get('lon')}")

    print("\n=== Aristas seleccionadas (x_e=1) ===")
    for (i,j,attr) in sorted(chosen_edges):
        print(f"({i},{j}) length={attr.get('length')} inter_component={attr.get('inter_component')} name={attr.get('name')}")

if __name__ == "__main__":
    main()
