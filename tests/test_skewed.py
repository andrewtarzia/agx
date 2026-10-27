import pathlib

import agx

from .case_data import CaseData


def test_skewed(skewed_graph_data: CaseData) -> None:
    """Test topology code enumeration.

    This should produce a graph and check the string against known ones, then
    delete them.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = (
        pathlib.Path(__file__).resolve().parent / "temp_skewed_graphs"
    )
    known_graph_directory = (
        pathlib.Path(__file__).resolve().parent / "test_skewed_graphs"
    )

    # Delete them.
    if graph_directory.exists():
        for filen in graph_directory.iterdir():
            filen.unlink()
        graph_directory.rmdir()

    graph_directory.mkdir(exist_ok=False)
    known_graph_directory.mkdir(exist_ok=True)

    iterator = agx.SkewedTopologyIterator(
        node_counts=skewed_graph_data.node_counts,
        max_samples=skewed_graph_data.max_samples,
        # Remake graphs.
        graph_directory=graph_directory,
    )
    assert iterator.count_graphs() == skewed_graph_data.num_graphs

    filename = graph_directory / f"skw_{iterator.graph_type}.json.gz"
    for tc in iterator.yield_graphs():
        # Look at previous string.
        str_file = (
            known_graph_directory / f"str_{iterator.graph_type}_{tc.idx}.txt"
        )
        if not str_file.exists():
            with str_file.open("w") as f:
                f.write(tc.get_as_string())

        with str_file.open("r") as f:
            lines = f.readlines()

        assert lines[0] == tc.get_as_string()

    # Delete them.
    filename = graph_directory / f"skw_{iterator.graph_type}.json.gz"
    filename.unlink()
    graph_directory.rmdir()
