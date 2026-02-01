Available graphs
================

A list of the graphs available in :mod:`agx`.
Files can be found in ``src/agx/_internal/known_graphs``.

New naming convention ``rxx``:
------------------------------

For the new ``graph_set``: ``rxx``, ``graph_type`` names are defined by
separating the building blocks of different number of FGs by underscores and
using hyphens to distinguish the multiplier by the number of FGs.

For example: ``rxx_4-4FG_6-2FG_4-1FG`` has 4 4FG building blocks,
6 2FG building blocks and 4 1FG building blocks.

Note that making new graphs is currently quite time consuming, so start with
a smaller ``max_samples`` than the default for ``rxx``, which is ``1e6``.

Graphs:
-------

.. important::

  All new graphs are run with a ``max_samples`` of 1e6.

Filtering graphs
^^^^^^^^^^^^^^^^

The code no longer hard codes filtering within generation, but
:class:`agx.TopologyCode` offers methods for filtering for
double wells or parallel edges.
Additionally, one might want to build graphs that are not only
``one connected graph``. To do so, you must change ``allowed_num_components``.


.. important::

  While we have one, two and three type graphs below, within each, any
  stoichiometry or configuration of same-numbered connections can be used.


One-type graphs
^^^^^^^^^^^^^^^

Produced graphs for ``m`` in (1 - 13) with FGs in (1 - 5).
Generated with code:

.. code-block:: python

    bbs = {1, 2, 3, 4, 5}
    multipliers = range(1, 11)
    for midx, fgnum in it.product(multipliers, bbs):
        string = f"{midx}-{fgnum}FG"
        iterator = agx.TopologyIterator(
            node_counts={
                agx.NodeType(type_id=0, num_connections=fgnum): midx,
            },
            graph_set="rxx",
        )
        if iterator.is_type_possible():
            logger.info("trying %s", string)
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )



.. testcode:: avail_graphs1-test
    :hide:

    import itertools as it
    import agx
    import sys

    bbs = {1, 2, 3, 4, 5}
    multipliers = range(1, 13)
    missing = set()
    for midx, fgnum in it.product(multipliers, bbs):
        iterator = agx.TopologyIterator(
            node_counts={
                agx.NodeType(type_id=0, num_connections=fgnum): midx,
            },
            graph_set="rxx",
        )

        if not iterator.is_type_possible():
            continue

        if not (
            iterator.graph_directory
            / f"rxx_{iterator.graph_type}.json.gz"
        ).exists():
            missing.add(iterator.graph_type)


    print(missing, file=sys.stderr)
    print(len(missing), file=sys.stderr)
    assert len(missing) == 4

Not all graphs have been built due to cost, or non-valid chemistry.

.. code-block:: python

    # Missing.
    [
        '1-1FG', '1-3FG', '1-5FG',
        '3-1FG', '3-3FG', '3-5FG',
        '5-1FG', '5-3FG', '5-5FG',
        '7-1FG', '7-3FG', '7-5FG',
        '9-1FG', '9-3FG', '9-5FG',
        '11-1FG', '11-3FG', '11-4FG', '11-5FG',
        '12-2FG', '12-3FG', '12-4FG', '12-5FG'
    ]

Two-type graphs
^^^^^^^^^^^^^^^

Produced graphs for ``m`` in (1 - 12) with FGs in (1 - 5) and
a combinatorial set of stoichiometries of ``bigger``:``smaller``.
We limit the multiplier for stoichiometries other than 1:2.
Generated with code:

.. code-block:: python

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
        if stoich not in ((1, 2), ) and midx > 4:  # noqa: PLR2004
            continue

        fgnum1_, fgnum2_ = sorted((fgnum1, fgnum2), reverse=True)

        string = (
            f"{midx * stoich[0]}-{fgnum1_}FG_"
            f"{midx * stoich[1]}-{fgnum2_}FG"
        )

        iterator = agx.TopologyIterator(
            node_counts={
                agx.NodeType(type_id=0, num_connections=fgnum1_): midx
                * stoich[0],
                agx.NodeType(type_id=1, num_connections=fgnum2_): midx
                * stoich[1],
            },
            graph_set="rxx",
        )
        if iterator.is_type_possible():
            logger.info("trying %s", string)
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )


.. testcode:: avail_graphs2-test
    :hide:

    import itertools as it
    import agx
    import sys

    bbs = {1, 2, 3, 4, 5}

    # Two typers.
    multipliers = range(1, 13)
    two_type_stoichiometries = tuple(
        (i, j) for i, j in it.product((1, 2, 3, 4, 5), repeat=2)
    )
    missing = set()
    for midx, fgnum1, fgnum2, stoich in it.product(
        multipliers, bbs, bbs, two_type_stoichiometries
    ):
        if fgnum1 == fgnum2:
            continue

        # Do not do all for larger stoichiomers.
        if stoich not in ((1, 2),) and midx > 4:  # noqa: PLR2004
            continue

        fgnum1_, fgnum2_ = sorted((fgnum1, fgnum2), reverse=True)

        iterator = agx.TopologyIterator(
            node_counts={
                agx.NodeType(type_id=0, num_connections=fgnum1_): midx
                * stoich[0],
                agx.NodeType(type_id=1, num_connections=fgnum2_): midx
                * stoich[1],
            },
            graph_set="rxx",
        )

        if not iterator.is_type_possible():
            continue

        if not (
            iterator.graph_directory
            / f"rxx_{iterator.graph_type}.json.gz"
        ).exists():
            missing.add(iterator.graph_type)

    print(missing, file=sys.stderr)
    print(len(missing), file=sys.stderr)
    assert len(missing) == 9


Not all graphs have been built due to cost, or non-valid chemistry.

.. code-block:: python

    # Missing.
    [
        '8-5FG_20-2FG', '8-5FG_10-4FG',
        '9-5FG_15-3FG', '9-4FG_12-3FG',
        '12-4FG_16-3FG', '12-5FG_15-4FG','12-5FG_20-3FG', '12-4FG_24-2FG',
        '16-5FG_20-4FG',
    ]




Three-type graphs
^^^^^^^^^^^^^^^^^

Produced graphs for ``m`` in (1, 2) with FGs in (1 - 4) and
a combinatorial check of stoichiometries. Note that current versions will
always focus on smaller FG BBs binding only to the BB with the most FGs.
Generated with code:

.. code-block:: python

    bbs = {1, 2, 3, 4, 5}

    # Three typers.
    multipliers = (1, 2)
    three_type_stoichiometries = tuple(
        (i, j, k) for i, j, k in it.product((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), repeat=3)
    )
    for midx, fgnum1, fgnum2, fgnum3, stoich in it.product(
        multipliers, bbs, bbs, bbs, three_type_stoichiometries
    ):
        if fgnum1 in (fgnum2, fgnum3) or fgnum2 == fgnum3:
            continue
        fgnum1_, fgnum2_, fgnum3_ = sorted(
            (fgnum1, fgnum2, fgnum3), reverse=True
        )

        string = (
            f"{midx * stoich[0]}-{fgnum1_}FG_"
            f"{midx * stoich[1]}-{fgnum2_}FG_"
            f"{midx * stoich[2]}-{fgnum3_}FG"
        )

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
        if iterator.is_type_possible():
            logger.info("trying %s", string)
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )


.. testcode:: avail_graphs3-test
    :hide:

    import itertools as it
    import agx
    import sys

    bbs = {1, 2, 3, 4, 5}

    # Three typers.
    multipliers = (1, 2)
    three_type_stoichiometries = tuple(
        (i, j, k) for i, j, k in it.product((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), repeat=3)
    )
    missing = set()
    for midx, fgnum1, fgnum2, fgnum3, stoich in it.product(
        multipliers, bbs, bbs, bbs, three_type_stoichiometries
    ):
        if fgnum1 in (fgnum2, fgnum3) or fgnum2 == fgnum3:
            continue
        fgnum1_, fgnum2_, fgnum3_ = sorted(
            (fgnum1, fgnum2, fgnum3), reverse=True
        )

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

        if not iterator.is_type_possible():
            continue

        if not (
            iterator.graph_directory
            / f"rxx_{iterator.graph_type}.json.gz"
        ).exists():
            missing.add(iterator.graph_type)

    print(missing, file=sys.stderr)
    print(len(missing), file=sys.stderr)
    assert len(missing) == 314

Not all graphs have been built due to cost, or non-valid chemistry.

.. code-block:: python

    # Missing.
    [
        "3-5FG_3-3FG_3-2FG",
        "4-5FG_4-4FG_2-2FG", "4-5FG_4-3FG_8-1FG", "4-5FG_8-2FG_4-1FG",
        "4-5FG_2-4FG_4-3FG", "4-5FG_6-3FG_2-1FG", "4-5FG_2-4FG_6-2FG",
        "4-5FG_3-4FG_4-2FG", "4-5FG_6-2FG_8-1FG", "4-4FG_2-3FG_5-2FG",
        "4-5FG_4-3FG_4-2FG",
        "5-5FG_4-4FG_3-3FG", "5-4FG_4-3FG_4-2FG", "5-5FG_5-3FG_5-2FG",
        "6-3FG_4-2FG_10-1FG", "6-4FG_10-2FG_4-1FG", "6-4FG_4-3FG_6-2FG",
        "6-5FG_6-4FG_6-1FG", "6-5FG_10-2FG_10-1FG", "6-5FG_6-3FG_6-2FG",
        "6-5FG_8-3FG_6-1FG", "6-5FG_6-4FG_2-3FG",
        "8-4FG_8-3FG_8-1FG", "8-4FG_10-3FG_2-1FG", "8-5FG_8-4FG_8-1FG",
        "8-3FG_10-2FG_4-1FG", "8-5FG_8-3FG_8-2FG", "8-4FG_8-3FG_4-2FG",
        "8-5FG_6-4FG_8-2FG", "8-5FG_10-3FG_10-1FG", "8-5FG_8-4FG_4-2FG",
        "8-5FG_4-4FG_8-3FG", "8-4FG_4-3FG_10-2FG",
        "10-5FG_10-4FG_10-1FG", "10-5FG_8-4FG_6-3FG", "10-4FG_10-3FG_10-1FG",
        "10-4FG_8-3FG_8-2FG", "10-5FG_10-3FG_10-2FG", "10-3FG_10-2FG_10-1FG",
        # And many more...
    ]
