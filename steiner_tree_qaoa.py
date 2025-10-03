"""Quantum Steiner tree solver for bikeway planning using QAOA.

This script reuses the existing OSM parsing utilities in the repository to build a
quadratic unconstrained binary optimization (QUBO) formulation of the Steiner tree
problem tailored to bikeway planning. Terminals represent mandatory points that must
be connected and pre-existing bikeways provide Steiner candidates that can shorten
construction cost. The QUBO is solved with Qiskit's QAOA implementation and the
resulting subgraph is visualised with NetworkX.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qiskit.primitives import Sampler
try:
    from qiskit.primitives import StatevectorSampler
except ImportError:  # Legacy Qiskit versions
    StatevectorSampler = None  # type: ignore[assignment]
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qaoa import OSMGraph, haversine_distance  # Reuse existing utilities


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverConfig:
    """Configuration parameters for the QUBO penalties."""

    lambda_edge_node: float = 12.0  # Penalises edge usage without activating endpoints
    lambda_terminal_deg: float = 18.0  # Enforces that terminals have at least one incident edge
    lambda_tree_balance: float = 6.0  # Encourages |E| = |V|-1 (tree structure)
    terminal_ids: Sequence[int] = ()
    leaf_terminal_ids: Sequence[int] = ()
    steiner_candidate_ids: Sequence[int] = ()
    steiner_activation_penalty: float = 0.4
    auxiliary_activation_penalty: float = 1.5
    cost_scaling: float = 1.0  # Multiplier for the metric edge cost


class SteinerQuboBuilder:
    """Builds a QuadraticProgram encoding for the Steiner tree bikeway problem."""

    def __init__(self, graph: nx.Graph, config: SolverConfig):
        self.G = graph
        self.config = config
        self.qp = QuadraticProgram("steiner_tree_bikeways")
        self.node_var_names: Dict[int, str] = {}
        self.edge_var_names: Dict[Tuple[int, int], str] = {}
        self.edge_costs: Dict[str, float] = {}
        self.linear_terms: Dict[str, float] = {}
        self.quadratic_terms: Dict[Tuple[str, str], float] = {}
        self.constant_offset: float = 0.0

        self.terminals: Set[int] = set(config.terminal_ids)
        self.leaf_terminals: Set[int] = set(config.leaf_terminal_ids)
        self.steiner_candidates: Set[int] = set(config.steiner_candidate_ids) - self.terminals
        self.all_nodes: List[int] = sorted(self.G.nodes())

        # Determine auxiliary nodes (not terminals nor preferred Steiner candidates)
        self.auxiliary_nodes: Set[int] = set(self.all_nodes) - self.terminals - self.steiner_candidates

    def build(self) -> QuadraticProgram:
        """Populate the QuadraticProgram with variables, objective and constraints."""
        self._add_node_variables()
        self._add_edge_variables_and_costs()
        self._initialise_objective_storage()
        self._add_edge_cost_component()
        self._add_node_activation_penalties()
        self._add_edge_node_consistency_penalty()
        self._add_terminal_connectivity_penalty()
        self._add_tree_balance_penalty()
        self._finalise_objective()
        self._add_terminal_activation_constraints()
        return self.qp

    # ------------------------------------------------------------------
    # Variable creation helpers
    # ------------------------------------------------------------------
    def _add_node_variables(self) -> None:
        for node_id in self.all_nodes:
            var_name = f"y_{node_id}"
            self.qp.binary_var(name=var_name)
            self.node_var_names[node_id] = var_name

    def _add_edge_variables_and_costs(self) -> None:
        for u, v, data in self.G.edges(data=True):
            key = self._edge_key(u, v)
            if key in self.edge_var_names:
                continue
            var_name = f"x_{key[0]}_{key[1]}"
            self.qp.binary_var(name=var_name)
            self.edge_var_names[key] = var_name
            self.edge_costs[var_name] = self._edge_cost(u, v, data)

    def _initialise_objective_storage(self) -> None:
        variables = list(self.node_var_names.values()) + list(self.edge_var_names.values())
        self.linear_terms = {name: 0.0 for name in variables}
        self.quadratic_terms = {}
        self.constant_offset = 0.0

    # ------------------------------------------------------------------
    # Objective components
    # ------------------------------------------------------------------
    def _add_edge_cost_component(self) -> None:
        for var_name, cost in self.edge_costs.items():
            self.linear_terms[var_name] += self.config.cost_scaling * cost

    def _add_node_activation_penalties(self) -> None:
        for node_id, var_name in self.node_var_names.items():
            if node_id in self.terminals:
                # Terminals are forced to be active, so no activation penalty needed
                continue
            penalty = (
                self.config.steiner_activation_penalty
                if node_id in self.steiner_candidates
                else self.config.auxiliary_activation_penalty
            )
            self.linear_terms[var_name] += penalty

    def _add_edge_node_consistency_penalty(self) -> None:
        weight = self.config.lambda_edge_node
        for (u, v), edge_var in self.edge_var_names.items():
            u_var = self.node_var_names[u]
            v_var = self.node_var_names[v]
            self._add_squared_penalty([(edge_var, 1.0), (u_var, -1.0)], weight)
            self._add_squared_penalty([(edge_var, 1.0), (v_var, -1.0)], weight)

    def _add_terminal_connectivity_penalty(self) -> None:
        weight = self.config.lambda_terminal_deg
        for terminal in self.terminals:
            incident_edge_vars = [
                self.edge_var_names[self._edge_key(terminal, nbr)]
                for nbr in self.G.neighbors(terminal)
                if self._edge_key(terminal, nbr) in self.edge_var_names
            ]
            if not incident_edge_vars:
                continue
            if terminal in self.leaf_terminals or len(incident_edge_vars) <= 1:
                target_degree = 1.0
            else:
                target_degree = 2.0
            expression = [(var, -1.0) for var in incident_edge_vars]
            self._add_squared_penalty(expression, weight, constant=target_degree)

    def _add_tree_balance_penalty(self) -> None:
        if not self.edge_var_names:
            return
        weight = self.config.lambda_tree_balance
        expression: List[Tuple[str, float]] = []
        expression.extend([(var_name, 1.0) for var_name in self.edge_var_names.values()])
        expression.extend([(var_name, -1.0) for var_name in self.node_var_names.values()])
        self._add_squared_penalty(expression, weight, constant=1.0)

    def _finalise_objective(self) -> None:
        self.qp.minimize(
            linear=self.linear_terms,
            quadratic=self.quadratic_terms,
            constant=self.constant_offset,
        )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def _add_terminal_activation_constraints(self) -> None:
        for terminal in self.terminals:
            var_name = self.node_var_names[terminal]
            self.qp.linear_constraint(
                linear={var_name: 1.0}, sense="==", rhs=1.0, name=f"terminal_active_{terminal}"
            )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _edge_cost(self, u: int, v: int, data: Dict) -> float:
        length = (
            data.get("length_m")
            or data.get("length")
            or data.get("length_meter")
        )
        if length is None:
            u_lat = self.G.nodes[u].get("lat")
            u_lon = self.G.nodes[u].get("lon")
            v_lat = self.G.nodes[v].get("lat")
            v_lon = self.G.nodes[v].get("lon")
            if None not in (u_lat, u_lon, v_lat, v_lon):
                length = haversine_distance(u_lat, u_lon, v_lat, v_lon)
            else:
                # Fallback to Euclidean distance in coordinate space
                length = float(np.hypot(u - v, u - v))
        return float(length)

    @staticmethod
    def _edge_key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    def _add_squared_penalty(
        self,
        expression_terms: Sequence[Tuple[str, float]],
        weight: float,
        constant: float = 0.0,
    ) -> None:
        if not expression_terms or weight == 0.0:
            if weight and constant:
                self.constant_offset += weight * constant * constant
            return

        self.constant_offset += weight * (constant ** 2)

        # Linear contributions from constant interaction
        for var, coeff in expression_terms:
            self.linear_terms[var] += weight * 2.0 * constant * coeff

        # Quadratic and diagonal contributions
        for i, (var_i, coeff_i) in enumerate(expression_terms):
            # Diagonal (i == j)
            self.linear_terms[var_i] += weight * coeff_i * coeff_i
            for j in range(i + 1, len(expression_terms)):
                var_j, coeff_j = expression_terms[j]
                pair = tuple(sorted((var_i, var_j)))
                self.quadratic_terms[pair] = (
                    self.quadratic_terms.get(pair, 0.0)
                    + weight * 2.0 * coeff_i * coeff_j
                )

    # ------------------------------------------------------------------
    # Solution interpretation helpers
    # ------------------------------------------------------------------
    def decode_solution(self, solution: Sequence[float]) -> Dict[str, object]:
        var_names = [var.name for var in self.qp.variables]
        assignment = dict(zip(var_names, solution))

        selected_nodes = {
            node for node, var in self.node_var_names.items() if assignment.get(var, 0.0) > 0.5
        }
        selected_edges = [
            (edge, self.edge_costs[var])
            for edge, var in self.edge_var_names.items()
            if assignment.get(var, 0.0) > 0.5
        ]

        total_cost = sum(cost for _, cost in selected_edges)

        return {
            "selected_nodes": selected_nodes,
            "selected_edges": selected_edges,
            "total_cost": total_cost,
            "assignment": assignment,
        }


def _default_terminal_nodes(osm_graph: OSMGraph) -> List[int]:
    preferred = [1001, 1003, 1007, 1008]
    if all(node in osm_graph.G for node in preferred):
        return preferred
    # Example terminals based on educational and transport POIs
    terminal_candidates: List[int] = []
    for node_id, data in osm_graph.G.nodes(data=True):
        tags = data.get("tags", {})
        if tags.get("name") in {"Start Point", "End Point"}:
            terminal_candidates.append(node_id)
        if tags.get("amenity") == "university":
            terminal_candidates.append(node_id)
        if tags.get("public_transport") == "bus_stop":
            terminal_candidates.append(node_id)
        if tags.get("highway") == "traffic_signal" and tags.get("name") in {"Start Point", "End Point"}:
            terminal_candidates.append(node_id)
    # Fallback: ensure at least two terminals
    if len(terminal_candidates) < 2:
        terminal_candidates.extend(list(osm_graph.G.nodes())[:2])
    # Deduplicate while preserving order
    seen: Set[int] = set()
    ordered: List[int] = []
    for node_id in terminal_candidates:
        if node_id not in seen:
            ordered.append(node_id)
            seen.add(node_id)
    return ordered


def _augment_isolated_terminals(graph: nx.Graph, terminals: Sequence[int]) -> None:
    coords: Dict[int, Tuple[float, float]] = {
        node: (data.get("lat"), data.get("lon"))
        for node, data in graph.nodes(data=True)
        if data.get("lat") is not None and data.get("lon") is not None
    }

    for terminal in terminals:
        if graph.degree(terminal) > 0:
            continue
        term_coords = coords.get(terminal)
        if term_coords is None:
            logger.warning(
                "No hay coordenadas válidas para el terminal aislado %s; no se puede crear arista candidata.",
                terminal,
            )
            continue

        best_node = None
        best_dist = float("inf")
        for node, node_coords in coords.items():
            if node == terminal:
                continue
            dist = haversine_distance(term_coords[0], term_coords[1], node_coords[0], node_coords[1])
            if dist < best_dist:
                best_dist = dist
                best_node = node

        if best_node is None or best_dist == float("inf"):
            logger.warning("No se encontró nodo vecino para conectar el terminal aislado %s.", terminal)
            continue

        key = (terminal, best_node)
        if graph.has_edge(*key):
            continue

        graph.add_edge(
            terminal,
            best_node,
            id=f"candidate_{terminal}_{best_node}",
            tags={
                "highway": "proposed",
                "bicycle": "yes",
                "existing_bikepath": "no",
                "is_candidate": "yes",
            },
            length_m=best_dist,
            width_m=3.0,
        )
        logger.info(
            "Arista candidata agregada para conectar el terminal aislado %s con %s (%.1f m)",
            terminal,
            best_node,
            best_dist,
        )


def build_solver_config(osm_graph: OSMGraph, terminals: Sequence[int] | None = None) -> SolverConfig:
    if terminals is None:
        terminals = _default_terminal_nodes(osm_graph)
    else:
        terminals = list(terminals)

    # Eliminar duplicados preservando el orden
    terminals = list(dict.fromkeys(terminals))

    leaf_terminals: List[int] = []
    if terminals:
        leaf_terminals.append(terminals[0])
    if len(terminals) > 1:
        leaf_terminals.append(terminals[-1])

    # Steiner candidates: nodes that belong to existing bikeways
    steiner_nodes: Set[int] = set()
    for u, v, data in osm_graph.G.edges(data=True):
        tags = data.get("tags", {})
        if tags.get("existing_bikepath") == "yes" or tags.get("bicycle") in {"designated", "yes"}:
            steiner_nodes.update([u, v])

    return SolverConfig(
        terminal_ids=terminals,
    leaf_terminal_ids=leaf_terminals,
        steiner_candidate_ids=sorted(steiner_nodes),
        lambda_edge_node=45.0,
        lambda_terminal_deg=650.0,
        lambda_tree_balance=80.0,
        steiner_activation_penalty=0.2,
        auxiliary_activation_penalty=0.6,
        cost_scaling=0.0005,
    )


def solve_steiner_tree(osm_json: Dict) -> Dict[str, object]:
    osm_graph = OSMGraph(osm_json)
    terminals = _default_terminal_nodes(osm_graph)
    _augment_isolated_terminals(osm_graph.G, terminals)
    # Recalcular terminales tras la posible adición de nuevas aristas
    config = build_solver_config(osm_graph, terminals)
    builder = SteinerQuboBuilder(osm_graph.G, config)
    qp = builder.build()

    if StatevectorSampler is not None:
        sampler = StatevectorSampler()
    else:
        logger.warning(
            "StatevectorSampler no disponible en esta versión de Qiskit; "
            "usando Sampler clásico."
        )
        sampler = Sampler()
    optimizer = COBYLA(maxiter=150)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2, initial_point=np.zeros(4))
    solver = MinimumEigenOptimizer(qaoa)

    logger.info("Starting QAOA optimization with %d variables", len(qp.variables))
    result = solver.solve(qp)
    logger.info("Optimization completed. Objective value: %.3f", result.fval)

    decoded = builder.decode_solution(result.x)
    decoded["result"] = result
    decoded["graph"] = osm_graph.G
    decoded["config"] = config
    decoded["builder"] = builder
    return decoded


def _plot_solution(solution: Dict[str, object]) -> None:
    G: nx.Graph = solution["graph"]
    builder: SteinerQuboBuilder = solution["builder"]
    selected_nodes: Set[int] = solution["selected_nodes"]
    selected_edges: List[Tuple[Tuple[int, int], float]] = solution["selected_edges"]

    terminals = set(builder.config.terminal_ids)
    steiner_candidates = set(builder.config.steiner_candidate_ids)

    pos = {
        node: (data.get("lon", 0.0), data.get("lat", 0.0))
        for node, data in G.nodes(data=True)
    }

    plt.figure(figsize=(8, 6))
    # Draw base network
    nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=1.0, alpha=0.6)

    # Highlight selected edges
    selected_edge_list = list(edge for edge, _ in selected_edges)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=selected_edge_list,
        width=3.0,
        edge_color="green",
    )

    # Terminal nodes (mandatory)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(terminals),
        node_size=160,
        node_color="#e74c3c",
        label="Terminales",
    )

    # Steiner nodes that were activated
    steiner_used = [node for node in selected_nodes if node in steiner_candidates and node not in terminals]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=steiner_used,
        node_size=140,
        node_color="#3498db",
        label="Steiner seleccionados",
    )

    # Auxiliary nodes that ended up used
    auxiliary_used = [
        node
        for node in selected_nodes
        if node not in terminals and node not in steiner_candidates
    ]
    if auxiliary_used:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=auxiliary_used,
            node_size=110,
            node_color="#9b59b6",
            label="Auxiliares",
        )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={node: str(node) for node in G.nodes},
        font_size=8,
    )

    plt.title("Árbol de Steiner optimizado para ciclovías")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    data_path = Path(__file__).with_name("data.json")
    with data_path.open("r", encoding="utf-8") as fh:
        osm_json = json.load(fh)

    solution = solve_steiner_tree(osm_json)

    print("\n--- Resumen del árbol de Steiner ---")
    print(f"Nodos seleccionados: {sorted(solution['selected_nodes'])}")
    print("Aristas seleccionadas:")
    for (u, v), cost in solution["selected_edges"]:
        print(f"  ({u}, {v}) -> {cost:.1f} m")
    print(f"Costo total aproximado: {solution['total_cost']:.1f} m")
    print(f"Valor objetivo QAOA: {solution['result'].fval:.3f}")

    _plot_solution(solution)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
