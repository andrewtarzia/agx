import argparse
import itertools as it
import logging

import agx

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        help="type to build (one, two, three)",
    )

    return parser.parse_args()


def one_type_function() -> None:
    """Build one-type graphs."""
    bbs = {1, 2, 3, 4, 5}
    multipliers = range(1, 11)
    for midx, fgnum in it.product(multipliers, bbs):
        try:
            string = f"{midx}-{fgnum}FG"
            logger.info("trying %s", string)
            iterator = agx.TopologyIterator(
                node_counts={
                    agx.NodeType(type_id=0, num_connections=fgnum): midx,
                },
                graph_set="rxx",
            )
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            pass


def two_type_function() -> None:
    """Build two-type graphs."""
    bbs = {1, 2, 3, 4, 5}

    # Two typers.
    multipliers = range(1, 13)
    two_type_stoichiometries = tuple(
        (i, j) for i, j in it.product((1, 2, 3, 4, 5), repeat=2)
    )
    for midx, fgnum1, fgnum2, stoich in it.product(
        multipliers, bbs, bbs, two_type_stoichiometries
    ):
        if fgnum1 == fgnum2:
            continue

        # Do not do all for larger stoichiomers.
        if stoich not in ((1, 2),) and midx > 4:  # noqa: PLR2004
            continue

        fgnum1_, fgnum2_ = sorted((fgnum1, fgnum2), reverse=True)

        try:
            string = (
                f"{midx * stoich[0]}-{fgnum1_}FG_"
                f"{midx * stoich[1]}-{fgnum2_}FG"
            )
            logger.info("trying %s", string)
            iterator = agx.TopologyIterator(
                node_counts={
                    agx.NodeType(type_id=0, num_connections=fgnum1_): midx
                    * stoich[0],
                    agx.NodeType(type_id=1, num_connections=fgnum2_): midx
                    * stoich[1],
                },
                graph_set="rxx",
            )
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            pass


def three_type_function() -> None:
    """Build three-type graphs."""
    bbs = {1, 2, 3, 4, 5}

    # Three typers.
    multipliers = (1, 2)
    three_type_stoichiometries = tuple(
        (i, j, k) for i, j, k in it.product((1, 2, 3, 4, 5), repeat=3)
    )
    for midx, fgnum1, fgnum2, fgnum3, stoich in it.product(
        multipliers, bbs, bbs, bbs, three_type_stoichiometries
    ):
        if fgnum1 in (fgnum2, fgnum3) or fgnum2 == fgnum3:
            continue
        fgnum1_, fgnum2_, fgnum3_ = sorted(
            (fgnum1, fgnum2, fgnum3), reverse=True
        )

        try:
            string = (
                f"{midx * stoich[0]}-{fgnum1_}FG_"
                f"{midx * stoich[1]}-{fgnum2_}FG_"
                f"{midx * stoich[2]}-{fgnum3_}FG"
            )
            logger.info("trying %s", string)
            iterator = agx.TopologyIterator(
                node_counts={
                    agx.NodeType(type_id=0, num_connections=fgnum1_): midx
                    * stoich[0],
                    agx.NodeType(type_id=1, num_connections=fgnum2_): midx
                    * stoich[1],
                    agx.NodeType(type_id=2, num_connections=fgnum3_): midx
                    * stoich[2],
                },
                graph_set="rxx",
            )
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )

        except (ZeroDivisionError, ValueError):
            pass


def main() -> None:
    args = _parse_args()

    if args.type == "one":
        one_type_function()
    elif args.type == "two":
        two_type_function()
    elif args.type == "three":
        three_type_function()
    else:
        msg = f"{args.type} type cannot be done so far"
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
