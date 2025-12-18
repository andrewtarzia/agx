"""Define the topology code class."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class Node:
    """Container for a node.

    Parameters:
        id:
            The id of the node.

        type_id:
            This defines a collection of nodes as equivalent.

        num_connections:
            The number of connections the node makes.

    """

    id: int
    type_id: int
    num_connections: int


@dataclass(eq=True, frozen=True)
class NodeType:
    """Container for a node types.

    Parameters:
        type_id:
            This defines a collection of nodes as equivalent.

        num_connections:
            The number of connections the node makes.

    """

    type_id: int
    num_connections: int
