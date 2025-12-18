"""Define classes for enumeration of graphs."""

import gzip
import itertools as it
import json
import logging
import pathlib
from collections import Counter, abc, defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import rustworkx as rx

from .configuration import Configuration
from .configuredcode import ConfiguredCode
from .node import Node, NodeType
from .topology_code import TopologyCode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TopologyIterator:
    """Iterate over graphs.

    .. important::

      **Warning**: Currently, the order of ``node_counts`` has to
      have the building block with the most connections first!

    Parameters:
        node_counts:
            Dictionary of :class:`agx.NodeType` and their count in the
            proposed graphs. Always put the building blocks with more
            functional groups first. Additionally, only mixtures of three
            distinct node types (in terms of connection counts) are
            implemented, and in the case of three components, all nodes connect
            to the node type with the most connections.

        graph_type:
            Name of the graph. If you do not need a custom graph, you can leave
            this as ``None`` and it will adhere to the current name convention
            (see ``available_graphs``), which captures the count of each node
            type with certain number of connections. Following this name
            convention will allow you to use saved graphs, if not, you can make
            your own. Although it can be time consuming.

        graph_set:
            Set of graphs to use based on different algorithms or papers.
            Only the new ``rxx`` set are defined here.

        allowed_num_components:
            Allowed number of disconnected graph components. Usually ``1`` to
            generate complete graphs only.

        max_samples:
            When constructing graphs, there is some randomness in their order,
            although that order should be consistent, and only up-to
            ``max_samples`` are sampled. For very large numbers of components
            there is not guarantee all possible graphs will be explored.

        graph_directory:
            Directory to check for and save graph jsons.

        verbose:
            Whether to log outcomes.

    """

    node_counts: dict[NodeType, int]
    graph_type: str | None = None
    graph_set: str = "rxx"
    allowed_num_components: int = 1
    max_samples: int | None = None
    graph_directory: pathlib.Path | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        """Initialize."""
        if self.graph_type is None:
            self.graph_type = self.generate_graph_type()

        if self.graph_directory is None:
            self.graph_directory = (
                pathlib.Path(__file__).resolve().parent / "known_graphs"
            )

        if not self.graph_directory.exists():
            msg = f"graph directory does not exist ({self.graph_directory})"
            raise RuntimeError(msg)

        self.graph_path = (
            self.graph_directory
            / f"{self.graph_set}_{self.graph_type}.json.gz"
        )
        if self.graph_set == "rxx":
            if self.max_samples is None:
                self.used_samples = int(1e6)
            else:
                self.used_samples = int(self.max_samples)

        elif self.max_samples is None:
            msg = (
                f"{self.graph_set} not defined, so you must set `max_samples`"
                " to make a new set."
            )
            raise NotImplementedError(msg)

        else:
            self.used_samples = int(self.max_samples)

        # Write vertex prototypes as a function of number of functional groups
        # and position them on spheres.
        vertex_prototypes: list[Node] = []
        reactable_vertex_ids = []
        num_edges = 0
        vertex_types_by_conn = defaultdict(list)
        for node_type, num_instances in self.node_counts.items():
            for _ in range(num_instances):
                vertex_id = len(vertex_prototypes)
                vertex_types_by_conn[node_type.num_connections].append(
                    vertex_id
                )
                num_edges += node_type.num_connections
                reactable_vertex_ids.extend(
                    [vertex_id] * node_type.num_connections
                )
                vertex_prototypes.append(
                    Node(
                        id=vertex_id,
                        type_id=node_type.type_id,
                        num_connections=node_type.num_connections,
                    )
                )

        self.vertex_types_by_conn = {
            i: tuple(vertex_types_by_conn[i]) for i in vertex_types_by_conn
        }
        self.reactable_vertex_ids = reactable_vertex_ids
        self.vertex_prototypes = vertex_prototypes
        self.vertex_counts = {
            i.id: i.num_connections for i in vertex_prototypes
        }

    def generate_graph_type(self) -> str:
        """Get the graph type to match the new naming convention."""
        fgcounts: dict[int, int] = defaultdict(int)
        for nodetype, count in self.node_counts.items():
            fgcounts[nodetype.num_connections] += count

        string = ""
        for fgtype, fgnum in sorted(fgcounts.items(), reverse=True):
            string += f"{fgnum}-{fgtype}FG_"

        return string.rstrip("_")

    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        return len(self.vertex_prototypes)

    def get_vertex_prototypes(self) -> abc.Sequence[Node]:
        """Get vertex prototypes."""
        return self.vertex_prototypes

    def _passes_tests(
        self,
        topology_code: TopologyCode,
        combinations_tested: set[str],
        combinations_passed: list[abc.Sequence[tuple[int, int]]],
    ) -> bool:
        # Need to check for nonsensical ones here.
        # Check the number of egdes per vertex is correct.
        counter = Counter([i for j in topology_code.vertex_map for i in j])
        if counter != self.vertex_counts:
            return False

        # If there are any self-reactions.
        if any(abs(i - j) == 0 for i, j in topology_code.vertex_map):
            return False

        # Check for string done.
        if topology_code.get_as_string() in combinations_tested:
            return False

        # Convert TopologyCode to a graph.
        current_graph = topology_code.get_graph()

        # Check that graph for isomorphism with other graphs.
        passed_iso = True
        for idx, tcc in enumerate(combinations_passed):
            test_graph = TopologyCode(idx, tcc).get_graph()

            if rx.is_isomorphic(current_graph, test_graph):
                passed_iso = False
                break

        return passed_iso

    def _one_type_algorithm(self) -> None:
        # All combinations tested.
        combinations_tested: set[str] = set()
        # All passed combinations.
        combinations_passed: list[abc.Sequence[tuple[int, int]]] = []

        type1 = next(iter(set(self.vertex_types_by_conn.keys())))

        rng = np.random.default_rng(seed=100)
        options = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_conn[type1]
        ]

        for i in range(self.used_samples):
            # Shuffle.
            rng.shuffle(options)
            # Split in half.
            half1 = options[: len(options) // 2]
            half2 = options[len(options) // 2 :]
            # Build an edge selection.
            try:
                combination: abc.Sequence[tuple[int, int]] = [
                    tuple(sorted((i, j)))  # type:ignore[misc]
                    for i, j in zip(half1, half2, strict=True)
                ]
            except ValueError as exc:
                msg = "could not split edge into two equal sets"
                raise ValueError(msg) from exc

            topology_code = TopologyCode(
                idx=len(combinations_passed),
                vertex_map=combination,
            )
            if self._passes_tests(
                topology_code=topology_code,
                combinations_tested=combinations_tested,
                combinations_passed=combinations_passed,
            ):
                combinations_passed.append(combination)

            # Add this anyway, either gets skipped, or adds the new one.
            combinations_tested.add(topology_code.get_as_string())
            # Progress.
            if i % 10000 == 0 and self.verbose:
                logger.info(
                    "done %s of %s (%s/100.0); found %s",
                    i,
                    self.used_samples,
                    round((i / self.used_samples) * 100, 1),
                    len(combinations_passed),
                )

        with gzip.open(str(self.graph_path), "w", 9) as f:
            f.write(json.dumps(combinations_passed).encode("utf8"))

    def _two_type_algorithm(self) -> None:
        # All combinations tested.
        combinations_tested: set[str] = set()
        # All passed combinations.
        combinations_passed: list[abc.Sequence[tuple[int, int]]] = []

        type1, type2 = sorted(self.vertex_types_by_conn.keys(), reverse=True)

        const = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_conn[type1]
        ]

        rng = np.random.default_rng(seed=100)
        options = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_conn[type2]
        ]
        for i in range(self.used_samples):
            rng.shuffle(options)
            # Build an edge selection.
            combination: abc.Sequence[tuple[int, int]] = [
                tuple(sorted((i, j)))  # type:ignore[misc]
                for i, j in zip(const, options, strict=True)
            ]

            topology_code = TopologyCode(
                idx=len(combinations_passed),
                vertex_map=combination,
            )
            if self._passes_tests(
                topology_code=topology_code,
                combinations_tested=combinations_tested,
                combinations_passed=combinations_passed,
            ):
                combinations_passed.append(combination)

            # Add this anyway, either gets skipped, or adds the new one.
            combinations_tested.add(topology_code.get_as_string())

            # Progress.
            if i % 10000 == 0 and self.verbose:
                logger.info(
                    "done %s of %s (%s/100.0); found %s",
                    i,
                    self.used_samples,
                    round((i / self.used_samples) * 100, 1),
                    len(combinations_passed),
                )

        with gzip.open(str(self.graph_path), "w", 9) as f:
            f.write(json.dumps(combinations_passed).encode("utf8"))

    def _three_type_algorithm(self) -> None:
        # All combinations tested.
        combinations_tested: set[str] = set()
        # All passed combinations.
        combinations_passed: list[abc.Sequence[tuple[int, int]]] = []

        type1, type2, type3 = sorted(
            self.vertex_types_by_conn.keys(), reverse=True
        )

        itera1 = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_conn[type1]
        ]

        rng = np.random.default_rng(seed=100)
        options1 = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_conn[type2]
        ]
        options2 = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_conn[type3]
        ]
        for i in range(self.used_samples):
            # Merging options1 and options2 because they both bind to itera.
            mixed_options = options1 + options2
            rng.shuffle(mixed_options)

            # Build an edge selection.
            combination: abc.Sequence[tuple[int, int]] = [
                tuple(sorted((i, j)))  # type:ignore[misc]
                for i, j in zip(itera1, mixed_options, strict=True)
            ]

            topology_code = TopologyCode(
                idx=len(combinations_passed),
                vertex_map=combination,
            )
            if self._passes_tests(
                topology_code=topology_code,
                combinations_tested=combinations_tested,
                combinations_passed=combinations_passed,
            ):
                combinations_passed.append(combination)

            # Add this anyway, either gets skipped, or adds the new one.
            combinations_tested.add(topology_code.get_as_string())

            # Progress.
            if i % 10000 == 0 and self.verbose:
                logger.info(
                    "done %s of %s (%s/100.0); found %s",
                    i,
                    self.used_samples,
                    round((i / self.used_samples) * 100, 1),
                    len(combinations_passed),
                )

        with gzip.open(str(self.graph_path), "w", 9) as f:
            f.write(json.dumps(combinations_passed).encode("utf8"))

    def _define_graphs(self) -> list[list[tuple[int, int]]]:
        if not self.graph_path.exists():
            # Check if .json exists.
            new_graph = self.graph_path.with_suffix("")
            if new_graph.exists():
                with new_graph.open("r") as f:
                    temp = json.load(f)
                with gzip.open(str(self.graph_path), "w", 9) as f:
                    f.write(json.dumps(temp).encode("utf8"))
                raise SystemExit

            if self.verbose:
                logger.info("%s not found, constructing!", self.graph_path)
            num_types = len(self.vertex_types_by_conn.keys())

            if num_types == 1:
                self._one_type_algorithm()
            elif num_types == 2:  # noqa: PLR2004
                self._two_type_algorithm()
            elif num_types == 3:  # noqa: PLR2004
                self._three_type_algorithm()
            else:
                msg = (
                    "Not implemented for mixtures of more than 3 distinct "
                    "FG numbers"
                )
                raise RuntimeError(msg)

        with gzip.open(str(self.graph_path), "r", 9) as f:
            return json.load(f)

    def count_graphs(self) -> int:
        """Count completely connected graphs in iteration."""
        count = 0
        for idx, combination in enumerate(self._define_graphs()):
            topology_code = TopologyCode(idx=idx, vertex_map=combination)

            num_components = topology_code.get_number_connected_components()
            if num_components == self.allowed_num_components:
                count += 1

        return count

    def yield_graphs(self) -> abc.Generator[TopologyCode]:
        """Get constructed molecules from iteration.

        Yields only graphs with the allowed number of components.
        """
        for idx, combination in enumerate(self._define_graphs()):
            topology_code = TopologyCode(idx=idx, vertex_map=combination)

            num_components = topology_code.get_number_connected_components()
            if num_components == self.allowed_num_components:
                yield topology_code

    def _get_modifiable_types(self) -> tuple[int]:
        """Get the connection counts with >1 equivalent nodes."""
        count_of_connection_types: dict[int, int] = defaultdict(int)
        for node in self.node_counts:
            count_of_connection_types[node.num_connections] += 1

        modifiable_types = tuple(
            connection_count
            for connection_count, count in count_of_connection_types.items()
            if count > 1
        )

        if len(modifiable_types) != 1:
            msg = (
                f"modifiable_types is len {len(modifiable_types)}. If 0"
                ", then you have no need to screen building block "
                "configurations. If greater than 2, then this code cannot "
                "handle this yet. Sorry!"
            )
            raise RuntimeError(msg)
        return modifiable_types

    def get_configurations(self) -> abc.Sequence[Configuration]:
        """Get potential node configurations."""
        # Get building blocks with the same functional group count - these are
        # swappable.
        modifiable_types = self._get_modifiable_types()

        # Get the associated vertex ids.
        modifiable_vertices = {
            num_connections: self.vertex_types_by_conn[num_connections]
            for num_connections in self.vertex_types_by_conn
            if num_connections in modifiable_types
        }

        unmodifiable_vertices = {
            num_connections: self.vertex_types_by_conn[num_connections]
            for num_connections in self.vertex_types_by_conn
            if num_connections not in modifiable_types
        }

        # Generate the configuration dictionary, filling in unmodifiable node
        # types, and giving an empty space for modifidable ones.
        empty_config_dict: dict[int, list[int]] = {}
        for node in self.node_counts:
            if node.num_connections in modifiable_types:
                empty_config_dict[node.type_id] = []
            else:
                empty_config_dict[node.type_id] = list(
                    unmodifiable_vertices[node.num_connections]
                )

        # Get the list of node type ids that are modifiable.
        modifiable_type_ids = tuple(
            type_id
            for type_id, vertices in empty_config_dict.items()
            if len(vertices) == 0
        )

        # Define a default list of the modifiable node indices to check new
        # configurations.
        modifiable_default = []
        for node, count in self.node_counts.items():
            if node.type_id not in modifiable_type_ids:
                continue
            modifiable_default.extend([node.type_id] * count)

        # Iterate over the placement of the bb indices.
        vertex_map = {
            v_idx: idx
            for idx, v_idx in enumerate(
                # ASSUMES 1 modifiable FG.
                modifiable_vertices[modifiable_types[0]]
            )
        }

        iteration = it.product(
            # ASSUMES 1 modifiable FG.
            *(
                modifiable_type_ids
                for i in modifiable_vertices[modifiable_types[0]]
            )
        )

        saved_dicts = set()
        possible_dicts: list[Configuration] = []
        for config_integers in iteration:
            # Check for default requirements for configuration.
            if sorted(config_integers) != modifiable_default:
                continue

            config_dict = {
                vertex_id: config_integers[vertex_map[vertex_id]]
                # ASSUMES 1 modifiable FG.
                for vertex_id in modifiable_vertices[modifiable_types[0]]
            }

            new_possibility = deepcopy(empty_config_dict)
            for vertex_id, node_type_id in config_dict.items():
                new_possibility[node_type_id].append(vertex_id)

            config = Configuration(
                idx=len(possible_dicts),
                node_types=self.node_counts,
                node_idx_dict={
                    i: tuple(j) for i, j in new_possibility.items()
                },
            )

            # Check for deduplication.
            if config.get_hashable_idx_dict() in saved_dicts:
                continue

            saved_dicts.add(config.get_hashable_idx_dict())
            possible_dicts.append(config)

        return possible_dicts

    def yield_configured_codes(self) -> abc.Iterator[ConfiguredCode]:
        """Get potential node configurations."""
        for config, code in it.product(
            self.get_configurations(), self.yield_graphs()
        ):
            yield ConfiguredCode(topology_code=code, configuration=config)
