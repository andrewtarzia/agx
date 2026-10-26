import pathlib
from collections import abc
from dataclasses import dataclass

import agx


@dataclass(slots=True, frozen=True)
class CaseData:
    """A test case."""

    node_counts: dict[agx.NodeType, int]
    graph_set: str
    graph_directory: pathlib.Path
    graph_filename: str
    num_graphs: int
    num_configs: int
    max_samples: int
    iso_pass: abc.Sequence[tuple[int, int]]
    name: str


@dataclass(slots=True, frozen=True)
class SkewedCaseData:
    """A test case."""

    node_counts: dict[agx.NodeType, int]
    max_samples: int
    num_graphs: int
    name: str
