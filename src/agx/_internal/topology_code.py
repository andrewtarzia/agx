"""Define the topology code class."""

import logging
from collections import Counter, abc
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import rustworkx as rx

logger = logging.getLogger(__name__)


@dataclass
class TopologyCode:
    """Naming convention for topology graphs."""

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
        graph_type: Literal["spring", "kamada", "spectral"],
        scale: float,
        topology_code: TopologyCode,
        iterator: TopologyIterator,
        bb_config: BuildingBlockConfiguration | None,
    ) -> stk.ConstructedMolecule:
        """Take a graph that considers all atoms, and get atom positions.

        The initial graph is generated with `stko.Network.init_from_molecule`.

        .. important::

        **Warning**: There is no guarantee the graph layout will give identical
        coordinates in multiple runs.

        Parameters:
            graph_type:
                Which networkx layout to use (of `spring`, `kamada`).

            scale:
                Scale factor to apply to eventual constructed molecule.

            topology_code:
                The code defining the topology graph.

            iterator:
                The `scram` algorithm used to generate the graph and configuration.

            bb_config:
                The configuration of building blocks on the graph.

        Returns:
            A constructed molecule at (0, 0, 0).

        """
        logger.warning(
            "Caution with this because it currently can change the cis/trans in "
            "m2l4"
        )

        constructed_molecule = try_except_construction(
            iterator=iterator,
            topology_code=topology_code,
            building_block_configuration=bb_config,
            vertex_positions=None,
        )

        stko_graph = stko.Network.init_from_molecule(constructed_molecule)
        if graph_type == "spring":
            nx_positions = nx.spring_layout(stko_graph.get_graph(), dim=3)
        elif graph_type == "kamada":
            nx_positions = nx.kamada_kawai_layout(
                stko_graph.get_graph(), dim=3
            )
        else:
            raise NotImplementedError
        pos_mat = np.array([nx_positions[i] for i in nx_positions])
        return constructed_molecule.with_position_matrix(
            pos_mat * float(scale)
        ).with_centroid(np.array((0.0, 0.0, 0.0)))

    def get_vertexset_molecule(
        graph_type: str | None,
        scale: float,
        topology_code: TopologyCode,
        iterator: TopologyIterator,
        bb_config: BuildingBlockConfiguration,
    ) -> stk.ConstructedMolecule:
        """Take a graph and genereate from graph vertex positions.

        .. important::

        **Warning**: There is no guarantee the graph layout will give identical
        coordinates in multiple runs.

        Parameters:
            graph_type:
                Which networkx layout to use (of `spring`,
                `kamada`, `spectral`).

            scale:
                Scale factor to apply to eventual constructed molecule.

            topology_code:
                The code defining the topology graph.

            iterator:
                The `scram` algorithm used to generate the graph and configuration.

            bb_config:
                The configuration of building blocks on the graph.

        Returns:
            A constructed molecule at (0, 0, 0).

        """
        if graph_type is None:
            return try_except_construction(
                iterator=iterator,
                topology_code=topology_code,
                building_block_configuration=bb_config,
                vertex_positions=None,
            ).with_centroid(np.array((0.0, 0.0, 0.0)))

        nx_graph = topology_code.get_nx_graph()

        if graph_type == "kamada":
            nxpos = nx.kamada_kawai_layout(nx_graph, dim=3)
        elif graph_type == "spring":
            nxpos = nx.spring_layout(nx_graph, dim=3)
        elif graph_type == "spectral":
            nxpos = nx.spectral_layout(nx_graph, dim=3)
        else:
            raise NotImplementedError

        vertex_positions = {
            nidx: np.array(nxpos[nidx]) * float(scale)
            for nidx in topology_code.get_nx_graph().nodes
        }
        return try_except_construction(
            iterator=iterator,
            topology_code=topology_code,
            building_block_configuration=bb_config,
            vertex_positions=vertex_positions,
        ).with_centroid(np.array((0.0, 0.0, 0.0)))
