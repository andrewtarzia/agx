"""Define the topology code class."""

import logging
from collections import Counter, abc
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
import rustworkx as rx

logger = logging.getLogger(__name__)


class ParallelEdgeError(Exception): ...


@dataclass
class TopologyCode:
    """Naming convention for topology graphs."""

    idx: int
    vertex_map: abc.Sequence[tuple[int, int]]

    def get_as_string(self) -> str:
        """Convert TopologyCode to string of the vertex map."""
        strs = sorted([f"{i[0]}-{i[1]}" for i in self.vertex_map])
        return "_".join(strs)

    def get_nauty_graph(self) -> pynauty.Graph:
        """Get nauty graph."""
        if self.contains_parallels():
            msg = "pynauty implementation requires no parallel edges."
            raise ParallelEdgeError(msg)

        graph = self.get_nx_graph()
        nodes = list(graph.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        adj = {idx[u]: [idx[v] for v in graph[u]] for u in nodes}
        types = {graph.nodes[n]["type"] for n in nodes}
        coloring = [
            {idx[n] for n in nodes if graph.nodes[n]["type"] == vertex_type}
            for vertex_type in types
        ]
        return pynauty.Graph(
            len(nodes),
            adjacency_dict=adj,
            vertex_coloring=coloring,
        )

    def get_nauty_certificate(self) -> bytes:
        """Exact canonical form: equal iff isomorphic."""
        return pynauty.certificate(self.get_nauty_graph())

    def get_nauty_automoprhism_grpsize(self) -> float:
        """Exact canonical form: equal iff isomorphic."""
        (
            # Automorphism basis set.
            _generators,
            grpsize1,
            _grpsize2,
            # The sets of equivalent vertices are called
            # orbits.
            _orbits,
            _numorbits,
        ) = pynauty.autgrp(self.get_nauty_graph())
        return grpsize1

    def get_nx_graph(self) -> nx.Graph | nx.MultiGraph:
        """Convert TopologyCode to a networkx graph."""
        graph = nx.MultiGraph() if self.contains_parallels() else nx.Graph()

        graph.add_edges_from(self.vertex_map)
        for n, d in graph.degree():
            graph.nodes[n]["type"] = d

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
        return self.contains_xs(4)

    def contains_xs(self, x: int) -> bool:
        """True if the graph contains cycles of size x.

        .. important::

            **Warning**: This does not currently handle repeated cycles,
            inverse cycles and shifted cycles.

        """
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
                # This does not currently handle repeated cycles, inverse
                # cycles and shifted cycles.
                if (
                    tuple(path) not in filtered_paths
                    and tuple(path[::-1]) not in filtered_paths
                ):
                    filtered_paths.add(tuple(path))

        path_lengths = [len(i) - 1 for i in filtered_paths]
        counter = Counter(path_lengths)

        return counter[x] != 0

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
        """Take a graph and generate a layout from graph vertex positions.

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
            nxpos = nx.spring_layout(nx_graph, dim=3, seed=100)
        elif layout_type == "spectral":
            nxpos = nx.spectral_layout(nx_graph, dim=3)
        else:
            raise NotImplementedError

        return {
            nidx: np.array(nxpos[nidx]) * float(scale)
            for nidx in self.get_nx_graph().nodes
        }

    def get_skewness_upper_bound(self) -> int:
        """Count the number of dges removed until planar."""
        if self.contains_parallels():
            msg = "skewness implementation requires no parallel edges."
            raise ParallelEdgeError(msg)

        graph = self.get_nx_graph()
        h_graph = nx.Graph()
        h_graph.add_nodes_from(self.get_nx_graph())
        removed = 0
        for e in graph.edges():
            h_graph.add_edge(*e)
            if not nx.check_planarity(h_graph)[0]:
                h_graph.remove_edge(*e)
                removed += 1
        return removed

    def visualise(self, name: str, output_file: str | pathlib.Path) -> None:
        """Visualise a graph."""
        g = self.get_nx_graph()

        fig, ax = plt.subplots(figsize=(3, 3))
        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            node_color="tab:blue",
            node_size=150,
            edgecolors="k",
        )
        nx.draw_networkx_labels(g, pos=pos, font_color="w")
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            alpha=0.4,
            edge_color="k",
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_title(name, fontsize=16)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_file, dpi=120, bbox_inches="tight")
        plt.close()

    def visualise_3d(self, name: str, output_file: str | pathlib.Path) -> None:
        """Visualise a graph."""
        colors = {
            2: "tab:gray",
            3: "tab:blue",
            4: "tab:orange",
        }

        g = self.get_nx_graph()

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(projection="3d")
        pos = nx.spring_layout(g, dim=3, seed=100)

        line_coord = []
        line_colors = []
        for edge in g.edges:
            e1_pos, e2_pos = pos[edge[0]], pos[edge[1]]
            e1_conn = g.nodes[edge[0]]["type"]
            e2_conn = g.nodes[edge[1]]["type"]
            # Every connection consist of two lines:
            # One from the first atom to the center
            # and from from the second atom to the center
            line_center = (e1_pos + e2_pos) / 2

            # Add line from first atom
            line_coord.append((e1_pos, line_center))
            line_colors.append(colors[e1_conn])

            # Add line from second atom
            line_coord.append((e2_pos, line_center))
            line_colors.append(colors[e2_conn])

        lines = Line3DCollection(
            line_coord,
            color=line_colors,
            linewidths=2,
            capstyle="round",
            joinstyle="round",
            zorder=-2,
        )
        ax.add_collection(lines)

        pos_mat = []
        for vid_, position in pos.items():
            pos_mat.append(position)
            # Plot balls
            ax.scatter(
                position[0],
                position[1],
                position[2],
                s=120,
                c=colors[g.nodes[vid_]["type"]],
                linewidth=1,
                alpha=1,
                zorder=2,
                ec="k",
            )
        pos_mat = np.array(pos_mat)  # type: ignore[assignment]
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_title(name, fontsize=16)
        ax.set_axis_off()
        # Set viewing angle
        ax.azim = -90
        ax.elev = 90

        # Make aspect ratio near 1.
        # from https://github.com/biotite-dev/biotite/blob/v2.0.0/src/biotite/structure/graphics/atoms.py#L13-L110
        zoom = 1
        resolved_center = (
            float((pos_mat[:, 0].max() + pos_mat[:, 0].min()) / 2),  # type: ignore[call-overload]
            float((pos_mat[:, 1].max() + pos_mat[:, 1].min()) / 2),  # type: ignore[call-overload]
            float((pos_mat[:, 2].max() + pos_mat[:, 2].min()) / 2),  # type: ignore[call-overload]
        )

        resolved_size = float(
            np.array(
                [
                    pos_mat[:, 0].max() - pos_mat[:, 0].min(),  # type: ignore[call-overload]
                    pos_mat[:, 1].max() - pos_mat[:, 1].min(),  # type: ignore[call-overload]
                    pos_mat[:, 2].max() - pos_mat[:, 2].min(),  # type: ignore[call-overload]
                ]
            ).max()
        )

        ax.set_xlim(
            resolved_center[0] - resolved_size / (2 * zoom),
            resolved_center[0] + resolved_size / (2 * zoom),
        )
        ax.set_ylim(
            resolved_center[1] - resolved_size / (2 * zoom),
            resolved_center[1] + resolved_size / (2 * zoom),
        )
        ax.set_zlim(
            resolved_center[2] - resolved_size / (2 * zoom),
            resolved_center[2] + resolved_size / (2 * zoom),
        )

        # Make the axis lengths of the 'plot box' equal
        # The 'plot box' is not visible due to 'axes.axis("off")'
        ax.set_box_aspect([1, 1, 1])
        fig.tight_layout()
        fig.savefig(output_file, dpi=120, bbox_inches="tight")
        plt.close()
