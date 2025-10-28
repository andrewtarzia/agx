"""Define the topology code class."""

import logging
from collections import Counter, abc
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
import rustworkx as rx

logger = logging.getLogger(__name__)


@dataclass
class TopologyCode:
    """Naming convention for topology graphs."""

    idx: int
    vertex_map: abc.Sequence[tuple[int, int]]

    def get_as_string(self) -> str:
        """Convert TopologyCode to string of the vertex map."""
        strs = sorted([f"{i[0]}-{i[1]}" for i in self.vertex_map])
        return "_".join(strs)

    def get_nx_graph(self) -> nx.Graph:
        """Convert TopologyCode to a networkx graph."""
        graph = nx.MultiGraph()

        for vert in self.vertex_map:
            graph.add_edge(vert[0], vert[1])

        return graph

    def get_graph(self) -> rx.PyGraph:
        """Convert TopologyCode to a graph."""
        graph: rx.PyGraph = rx.PyGraph(multigraph=True)

        vertices = {
            vi: graph.add_node(vi)
            for vi in sorted({i for j in self.vertex_map for i in j})
        }

        for vert in self.vertex_map:
            nodea = graph[vertices[vert[0]]]
            nodeb = graph[vertices[vert[1]]]
            graph.add_edge(nodea, nodeb, None)

        return graph

    def get_weighted_graph(self) -> rx.PyGraph:
        """Convert TopologyCode to a graph."""
        graph: rx.PyGraph = rx.PyGraph(multigraph=False)

        vertices = {
            vi: graph.add_node(vi)
            for vi in sorted({i for j in self.vertex_map for i in j})
        }

        for vert in self.vertex_map:
            nodea = graph[vertices[vert[0]]]
            nodeb = graph[vertices[vert[1]]]
            if not graph.has_edge(nodea, nodeb):
                graph.add_edge(nodea, nodeb, 1)
            else:
                graph.add_edge(
                    nodea, nodeb, graph.get_edge_data(nodea, nodeb) + 1
                )

        return graph

    def contains_doubles(self) -> bool:
        """True if the graph contains "double-walls"."""
        weighted_graph = self.get_weighted_graph()

        filtered_paths = set()
        for node in weighted_graph.nodes():
            paths = list(
                rx.graph_all_simple_paths(
                    weighted_graph,
                    origin=node,  # type: ignore[call-arg]
                    to=node,  # type: ignore[call-arg]
                    cutoff=12,
                    min_depth=4,
                )
            )

            for path in paths:
                if (
                    tuple(path) not in filtered_paths
                    and tuple(path[::-1]) not in filtered_paths
                ):
                    filtered_paths.add(tuple(path))

        path_lengths = [len(i) - 1 for i in filtered_paths]
        counter = Counter(path_lengths)

        return counter[4] != 0

    def contains_parallels(self) -> bool:
        """True if the graph contains "1-loops"."""
        weighted_graph = self.get_weighted_graph()
        num_parallel_edges = len([i for i in weighted_graph.edges() if i > 1])

        return num_parallel_edges != 0

    def get_number_connected_components(self) -> int:
        """Get the number of connected components."""
        return rx.number_connected_components(self.get_graph())

    def get_layout(
        self,
        layout_type: str,
        scale: float,
    ) -> dict[int, npt.NDArray[np.float64]]:
        """Take a graph and genereate from graph vertex positions.

        .. important::

            **Warning**: There is no guarantee the graph layout will give
            identical coordinates in multiple runs.

        Parameters:
            layout_type:
                Which networkx layout to use (of `spring`,
                `kamada`, `spectral`).

            scale:
                Scale factor to apply to eventual constructed molecule.

        Returns:
            Vertex positions.

        """
        nx_graph = self.get_nx_graph()
        if layout_type == "kamada":
            nxpos = nx.kamada_kawai_layout(nx_graph, dim=3)
        elif layout_type == "spring":
            nxpos = nx.spring_layout(nx_graph, dim=3)
        elif layout_type == "spectral":
            nxpos = nx.spectral_layout(nx_graph, dim=3)
        else:
            raise NotImplementedError

        return {
            nidx: np.array(nxpos[nidx]) * float(scale)
            for nidx in self.get_nx_graph().nodes
        }
