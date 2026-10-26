"""Utilities module."""

import gzip
import json
import logging
import pathlib
from collections import abc

from .configuredcode import ConfiguredCode

logger = logging.getLogger(__name__)


def is_configured_code_isomorphic(
    test_code: ConfiguredCode,
    run_topology_codes: abc.Sequence[ConfiguredCode],
) -> bool:
    """Check if a graph and bb config passes isomorphism check."""
    # Check that graph for isomorphism with others graphs.
    passed_iso = True
    for temp_code in run_topology_codes:
        if test_code.is_isomorphic(temp_code):
            passed_iso = False
            break
    return passed_iso


def load_from_json_gz(path: pathlib.Path) -> list[list[tuple[int, int]]]:
    """Load graphs from an archive."""
    with gzip.open(str(path), "r", 9) as f:
        return json.load(f)


def restart_to_finished(
    path_gzip: pathlib.Path,
    output_gzip: pathlib.Path,
) -> None:
    """Convert a restart file to a final file for testing."""
    with gzip.open(str(path_gzip), "r", 9) as f:
        dictionary = json.load(f)
        combos = dictionary["combos"]

    with gzip.open(str(output_gzip), "w", 9) as f:
        f.write(json.dumps(combos).encode("utf8"))
