import pathlib

import numpy as np

import agx

from .case_data import CaseData


def test_layouts(
    graph_data: CaseData,
    layout_type: str,
    scale: int,
) -> None:
    """Test graph layout processes."""
    known_mols = pathlib.Path(__file__).resolve().parent / "test_molecules"
    known_mols.mkdir(exist_ok=True)

    iterator = agx.TopologyIterator(
        node_counts=graph_data.node_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=pathlib.Path(__file__).resolve().parent
        / "test_graphs",
    )

    for tc in iterator.yield_graphs():
        vs_name = (
            known_mols
            / f"vs_{iterator.graph_type}_{tc.idx}_{layout_type}_{scale}.npy"
        )

        if not vs_name.exists():
            # Build it.
            coordinates = np.array(
                list(
                    tc.get_layout(
                        layout_type=layout_type, scale=scale
                    ).values()
                )
            )
            np.save(file=vs_name, arr=coordinates)

        test_coordinates = np.array(
            list(tc.get_layout(layout_type=layout_type, scale=scale).values())
        )
        test = np.divide(
            test_coordinates.sum(axis=0),
            len(test_coordinates),
        )

        known_coordinates = np.load(vs_name)
        know = np.divide(
            known_coordinates.sum(axis=0),
            len(known_coordinates),
        )

        # Actual coordinates are not constant, so test other measures.
        assert np.allclose(know, test, atol=1e-3)
