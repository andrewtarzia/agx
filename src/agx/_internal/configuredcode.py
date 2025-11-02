"""Module for comparing graphs."""

import logging
import typing
from dataclasses import dataclass

import rustworkx as rx

from .configuration import Configuration
from .topology_code import TopologyCode

logger = logging.getLogger(__name__)


@dataclass
class ConfiguredCode:
    """Naming convention for topology codes with configurations.

    Parameters:
        topology_code:
            The graph connecting the nodes.

        configuration:
            The symmetry, or arrangement of different nodes on the graph.

    """

    topology_code: TopologyCode
    configuration: Configuration

    def get_graph(self) -> rx.PyGraph:
        """Convert TopologyCode and Configuration to rx graph."""
        graph: rx.PyGraph = rx.PyGraph(multigraph=True)

        vertices = {}
        for vi in sorted(
            {i for j in self.topology_code.vertex_map for i in j}
        ):
            bb_id = next(
                i
                for i, vert_ids in self.configuration.node_idx_dict.items()
                if vi in vert_ids
            )

            vertices[f"{vi}-{bb_id}"] = graph.add_node(f"{vi}-{bb_id}")

        for vert in self.topology_code.vertex_map:
            v1 = vert[0]
            bb_id = next(
                i
                for i, vert_ids in self.configuration.node_idx_dict.items()
                if v1 in vert_ids
            )
            v1str = f"{v1}-{bb_id}"
            v2 = vert[1]
            bb_id = next(
                i
                for i, vert_ids in self.configuration.node_idx_dict.items()
                if v2 in vert_ids
            )
            v2str = f"{v2}-{bb_id}"
            nodeaidx = vertices[v1str]
            nodebidx = vertices[v2str]
            graph.add_edge(nodeaidx, nodebidx, None)

        return graph

    def is_isomorphic(
        self,
        other_configured_code: typing.Self,
    ) -> bool:
        """Check if two configured codes are isomorphic."""
        # Testing bb-config aware graph check.
        # Convert TopologyCode to a graph.
        current_graph = self.get_graph()

        test_graph = other_configured_code.get_graph()

        return rx.is_isomorphic(
            current_graph,
            test_graph,
            node_matcher=lambda x, y: x.split("-")[1] == y.split("-")[1],
        )
