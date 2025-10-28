"""Define classes for enumeration of graphs."""

import gzip
import json
import logging
import pathlib
from collections import Counter, abc, defaultdict
from dataclasses import dataclass

import numpy as np
import rustworkx as rx

from .node import Node, NodeType
from .topology_code import TopologyCode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TopologyIterator:
    """Iterate over topology graphs.

    This is the latest version, but without good symmetry and graph checks,
    this can over produce structures.

    .. important::

      **Warning**: Currently, the order of ``building_block_counts`` has to
      have the building block with the most FGs first! This ordering is defined
      by the order used when defining the graphs. If you are defining your own
      graph library (i.e., setting ``graph_directory`` or using a new
      ``graph_type``), then the order is defined by the order in
      ``building_block_counts`` when generating the json.

    .. important::

      To reproduce the ``no_doubles'' dataset, you must use
      ``graph_set=rx_nodoubles``, or filter the topology codes after
      generation using the :class:`cgexplore.scram.TopologyCode` methods
      (this is now the recommended approach).

    Parameters:
        building_block_counts:
            Dictionary of :class:`stk.BuildingBlock` and their count in the
            proposed structures. Always put the building blocks with more
            functional groups first (this is a current bug). Additionally, only
            mixtures of three distinct building block functional group counts
            is implemented, and in the case of three components, all building
            blocks bind to the building block with the most functional groups.

        graph_type:
            Name of the graph. Current name convention is long, but complete,
            capturing the count of each building block with certain functional
            group count included. Following this name convention will allow you
            to use saved graphs, if not, you can make your own. Although it can
            be time consuming.

        graph_set:
            Set of graphs to use based on different algorithms or papers.
            Can be custom, as above. Note that the code to generation ``nx``
            graphs is no longer present in ``cgexplore`` because the
            :mod:`networkx` algorithms were slow.

        scale_multiplier:
            Scale multiplier to use in construction.

        allowed_num_components:
            Allowed number of disconnected graph components. Usually ``1`` to
            generate complete graphs only.

        max_samples:
            When constructing graphs, there is some randomness in their order,
            although that order should be consistent, and only up-to
            ``max_samples`` are sampled. For very large numbers of building
            blocks there is not guarantee all possible graphs will be explored.

        graph_directory:
            Directory to check for and save graph jsons.

    """

    node_counts: dict[NodeType, int]
    graph_type: str
    graph_set: str = "rxx"
    scale_multiplier = 5
    allowed_num_components: int = 1
    max_samples: int | None = None
    graph_directory: pathlib.Path | None = None

    def __post_init__(self) -> None:
        """Initialize."""
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

    def get_num_nodes(self) -> int:
        """Get number of building blocks."""
        return len(self.vertex_prototypes)

    def get_vertex_prototypes(self) -> abc.Sequence[Node]:
        """Get vertex prototypes."""
        return self.vertex_prototypes

    def _passes_tests(
        self,
        topology_code: TopologyCode,
        combinations_tested: set,
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
            if i % 10000 == 0:
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
            if i % 10000 == 0:
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
            if i % 10000 == 0:
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

        Yields only completely connected graphs.
        """
        for idx, combination in enumerate(self._define_graphs()):
            topology_code = TopologyCode(idx=idx, vertex_map=combination)

            num_components = topology_code.get_number_connected_components()
            if num_components == self.allowed_num_components:
                yield topology_code
