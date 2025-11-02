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

Produced graphs for ``m`` in (1 - 11) with FGs in (1 - 5).
Generated with code:

.. code-block:: python

    bbs = {1, 2, 3, 4, 5}
    multipliers = range(1, 11)
    for midx, fgnum in it.product(multipliers, bbs):
        try:
            string = f"{midx}-{fgnum}FG"
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


.. testcode:: avail_graphs1-test
    :hide:

    import itertools as it
    import agx

    knowns = (
        "8-3FG",
        "8-4FG",
        "9-2FG",
        "9-4FG",
        "6-4FG",
        "7-2FG",
        "7-4FG",
        "8-1FG",
        "8-2FG",
        "2-4FG",
        "3-2FG",
        "3-4FG",
        "4-1FG",
        "4-2FG",
        "4-3FG",
        "4-4FG",
        "5-2FG",
        "5-4FG",
        "6-1FG",
        "6-2FG",
        "6-3FG",
        "1-4FG",
        "2-1FG",
        "2-2FG",
        "2-3FG",
        "1-2FG",
        "10-1FG",
        "10-2FG",
        "10-3FG",
        "10-4FG",
        "11-2FG",
        "12-1FG",
    )
    bbs = {1, 2, 3, 4, 5}
    multipliers = range(1, 11)
    for midx, fgnum in it.product(multipliers, bbs):
        try:
            iterator = agx.TopologyIterator(
                node_counts={
                    agx.NodeType(type_id=0, num_connections=fgnum): midx,
                },
                graph_set="rxx",
            )

            if iterator.graph_type in knowns:
                assert (
                    iterator.graph_directory / f"rxx_{iterator.graph_type}.json.gz"
                ).exists()

        except (ZeroDivisionError, ValueError):
            pass

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


.. testcode:: avail_graphs2-test
    :hide:

    import itertools as it
    import agx

    knowns = (
        "1-2FG_2-1FG",
        "1-4FG_2-2FG",
        "2-2FG_4-1FG",
        "2-3FG_3-2FG",
        "2-4FG_4-2FG",
        "3-4FG_4-3FG",
        "4-3FG_6-2FG",
    )

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
            iterator = agx.TopologyIterator(
                node_counts={
                    agx.NodeType(type_id=0, num_connections=fgnum1_): midx
                    * stoich[0],
                    agx.NodeType(type_id=1, num_connections=fgnum2_): midx
                    * stoich[1],
                },
                graph_set="rxx",
            )

            if iterator.graph_type in knowns:
                assert (
                    iterator.graph_directory / f"rxx_{iterator.graph_type}.json.gz"
                ).exists()

        except (ZeroDivisionError, ValueError):
            pass

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


.. testcode:: avail_graphs3-test
    :hide:

    import itertools as it
    import agx

    knowns = (
        "1-3FG_1-2FG_1-1FG",
        "1-4FG_1-2FG_2-1FG",
        "1-4FG_1-3FG_1-1FG",
        "2-3FG_1-2FG_4-1FG",
        "2-3FG_2-2FG_2-1FG",
        "2-4FG_2-2FG_4-1FG",
        "2-4FG_2-3FG_1-2FG",
        "2-4FG_2-3FG_2-1FG",
        "2-4FG_3-2FG_2-1FG",
        "3-3FG_3-2FG_3-1FG",
        "3-3FG_4-2FG_1-1FG",
        "3-4FG_2-3FG_3-2FG",
        "3-4FG_3-3FG_3-1FG",
        "3-4FG_4-2FG_4-1FG",
        "4-3FG_2-2FG_8-1FG",
        "4-3FG_4-2FG_4-1FG",
        "4-4FG_4-3FG_2-2FG",
        "4-4FG_4-3FG_4-1FG",
        "6-3FG_6-2FG_6-1FG",
        "6-3FG_8-2FG_2-1FG",
    )

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

            if iterator.graph_type in knowns:
                assert (
                    iterator.graph_directory / f"rxx_{iterator.graph_type}.json.gz"
                ).exists()

        except (ZeroDivisionError, ValueError):
            pass
