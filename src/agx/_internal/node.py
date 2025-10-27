"""Define the topology code class."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class Node:
    """Container for a node."""

    id: int
    num_connections: int


@dataclass(eq=True, frozen=True)
class NodeType:
    """Container for a node types."""

    num_connections: int
