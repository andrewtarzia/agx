"""Utilities module."""

import logging
from collections import abc

from .configuredcode import ConfiguredCode

logger = logging.getLogger(__name__)


def is_configured_code_isomoprhic(
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
