"""Script to generate and optimise CG models."""

import logging
from collections import abc
from dataclasses import dataclass

from .node import Node

logger = logging.getLogger(__name__)


@dataclass
class Configuration:
    """Naming convention for graph node configurations."""

    idx: int
    node_counts: dict[Node, int]
    node_idx_dict: dict[int, abc.Sequence[int]]

    def get_node_dictionary(self) -> dict[Node, abc.Sequence[int]]:
        """Get the node dictionary."""
        idx_map = {node.type_id: node for node in self.node_counts}
        return {
            idx_map[idx]: tuple(vertices)
            for idx, vertices in self.node_idx_dict.items()
        }

    def get_hashable_idx_dict(
        self,
    ) -> abc.Sequence[tuple[int, abc.Sequence[int]]]:
        """Get a hashable representation of the dictionary."""
        return tuple(sorted(self.node_idx_dict.items()))

    def __str__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return (
            f"{self.__class__.__name__}(idx={self.idx}, "
            f"node_idx_dict={self.node_idx_dict})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return str(self)
