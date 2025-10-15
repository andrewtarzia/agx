"""Script to generate and optimise CG models."""

import itertools as it
import logging
from collections import abc, defaultdict
from copy import deepcopy
from dataclasses import dataclass

import stk

from agx._internal.enumeration import TopologyIterator

logger = logging.getLogger(__name__)


@dataclass
class BuildingBlockConfiguration:
    """Naming convention for building block configurations."""

    idx: int
    building_block_idx_map: dict[stk.BuildingBlock, int]
    building_block_idx_dict: dict[int, abc.Sequence[int]]

    def get_building_block_dictionary(
        self,
    ) -> dict[stk.BuildingBlock, abc.Sequence[int]]:
        idx_map = {idx: bb for bb, idx in self.building_block_idx_map.items()}
        return {
            idx_map[idx]: tuple(vertices)
            for idx, vertices in self.building_block_idx_dict.items()
        }

    def get_hashable_bbidx_dict(
        self,
    ) -> abc.Sequence[tuple[int, abc.Sequence[int]]]:
        """Get a hashable representation of the building block dictionary."""
        return tuple(sorted(self.building_block_idx_dict.items()))

    def __str__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return (
            f"{self.__class__.__name__}(idx={self.idx}, "
            f"building_block_idx_dict={self.building_block_idx_dict})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return str(self)


def get_custom_bb_configurations(  # noqa: C901
    iterator: TopologyIterator,
) -> abc.Sequence[BuildingBlockConfiguration]:
    """Get potential building block dictionaries."""
    # Get building blocks with the same functional group count - these are
    # swappable.
    building_blocks_by_fg = {
        i: i.get_num_functional_groups() for i in iterator.building_blocks
    }

    count_of_fg_types: dict[int, int] = defaultdict(int)
    fg_counts_by_building_block: dict[int, int] = defaultdict(int)

    for bb, count in iterator.building_block_counts.items():
        fg_counts_by_building_block[bb.get_num_functional_groups()] += count
        count_of_fg_types[bb.get_num_functional_groups()] += 1

    modifiable_types = tuple(
        fg_count for fg_count, count in count_of_fg_types.items() if count > 1
    )
    if len(modifiable_types) != 1:
        msg = (
            f"modifiable_types is len {len(modifiable_types)}. If 0"
            ", then you have no need to screen building block configurations."
            " If greater than 2, then this code cannot handle this yet. Sorry!"
        )
        raise RuntimeError(msg)

    # Get the associated vertex ids.
    modifiable_vertices = {
        fg_count: iterator.vertex_types_by_fg[fg_count]
        for fg_count in iterator.vertex_types_by_fg
        # ASSUMES 1 modifiable FG.
        if fg_count == modifiable_types[0]
    }

    unmodifiable_vertices = {
        fg_count: iterator.vertex_types_by_fg[fg_count]
        for fg_count in iterator.vertex_types_by_fg
        # ASSUMES 1 modifiable FG.
        if fg_count != modifiable_types[0]
    }

    # Count of functional groups: number of vertices that need adding.
    count_to_add = {
        i: fg_counts_by_building_block[i] for i in modifiable_types
    }

    if len(count_to_add) != 1:
        msg = (
            f"count to add is len {len(count_to_add)}. If greater than 1, "
            "then this code cannot handle this yet. Sorry!"
        )
        raise RuntimeError(msg)

    bb_map = {bb: idx for idx, bb in enumerate(building_blocks_by_fg)}

    empty_bb_dict: dict[int, list[int]] = {}
    for bb, fg_count in building_blocks_by_fg.items():
        if fg_count in modifiable_types:
            empty_bb_dict[bb_map[bb]] = []
        else:
            empty_bb_dict[bb_map[bb]] = list(unmodifiable_vertices[fg_count])

    # ASSUMES 1 modifiable FG.
    modifiable_bb_idx = tuple(
        bb_idx
        for bb_idx, vertices in empty_bb_dict.items()
        if len(vertices) == 0
    )
    modifiable_bb_idx_counted = []
    for bb, count in iterator.building_block_counts.items():
        idx = bb_map[bb]
        if idx not in modifiable_bb_idx:
            continue
        modifiable_bb_idx_counted.extend([idx] * count)

    # Iterate over the placement of the bb indices.
    vertex_map = {
        v_idx: idx
        for idx, v_idx in enumerate(modifiable_vertices[modifiable_types[0]])
    }
    iteration = it.product(
        # ASSUMES 1 modifiable FG.
        *(modifiable_bb_idx for i in modifiable_vertices[modifiable_types[0]])
    )

    saved_bb_dicts = set()
    possible_dicts: list[BuildingBlockConfiguration] = []

    for config in iteration:
        if sorted(config) != modifiable_bb_idx_counted:
            continue

        bb_config_dict = {
            vertex_id: config[vertex_map[vertex_id]]
            for vertex_id in modifiable_vertices[modifiable_types[0]]
        }

        new_possibility = deepcopy(empty_bb_dict)
        for vertex_id, bb_idx in bb_config_dict.items():
            new_possibility[bb_idx].append(vertex_id)

        bbconfig = BuildingBlockConfiguration(
            idx=len(possible_dicts),
            building_block_idx_map=bb_map,
            building_block_idx_dict={
                i: tuple(j) for i, j in new_possibility.items()
            },
        )

        if bbconfig.get_hashable_bbidx_dict() in saved_bb_dicts:
            continue

        # Check for deduplication.
        saved_bb_dicts.add(bbconfig.get_hashable_bbidx_dict())

        possible_dicts.append(bbconfig)

    msg = (
        "bring rmsd checker in here: use symmetry corrected RMSD on "
        "single-bead repr of tstr"
    )
    logger.info(msg)

    return tuple(possible_dicts)
