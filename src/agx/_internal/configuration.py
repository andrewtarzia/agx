"""Script to generate and optimise CG models."""

import logging
from collections import abc
from dataclasses import dataclass

from .node import NodeType

logger = logging.getLogger(__name__)


@dataclass
class Configuration:
    """Naming convention for graph node configurations."""

    idx: int
    node_types: dict[NodeType, int]
    node_idx_dict: dict[int, abc.Sequence[int]]

    def get_node_dictionary(self) -> dict[NodeType, abc.Sequence[int]]:
        """Get the node dictionary.

        This is equivalent to the building block dictionary you need, where
        each key is a type of building block with a set number of connections.

        """
        idx_map = {
            node_type.type_id: node_type for node_type in self.node_types
        }
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
