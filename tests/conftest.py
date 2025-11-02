import pathlib

import pytest

import agx

from .case_data import CaseData

bbs = {1, 2, 3, 4}


@pytest.fixture(
    params=(
        lambda name: CaseData(
            node_counts={
                agx.NodeType(type_id=0, num_connections=4): 2,
                agx.NodeType(type_id=1, num_connections=2): 4,
            },
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            graph_filename="rxx_2-4FG_4-2FG.json.gz",
            num_graphs=2,
            num_configs=0,
            name=name,
            doubles={0: True, 1: True},
            parallels={0: True, 1: False},
            iso_pass=(),
        ),
        lambda name: CaseData(
            node_counts={
                agx.NodeType(type_id=0, num_connections=4): 3,
                agx.NodeType(type_id=1, num_connections=2): 3,
                agx.NodeType(type_id=2, num_connections=2): 3,
            },
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            graph_filename="rxx_3-4FG_6-2FG.json.gz",
            num_graphs=4,
            num_configs=20,
            name=name,
            doubles={0: True, 1: True, 2: True, 4: False},
            parallels={0: True, 1: True, 2: False, 4: True},
            iso_pass=(
                (0, 0),
                (1, 0),
                (2, 0),
                (4, 0),
                (1, 1),
                (2, 1),
                (4, 1),
                (1, 2),
                (4, 2),
                (0, 3),
                (0, 4),
                (4, 4),
                (1, 5),
                (4, 6),
                (1, 7),
                (0, 10),
                (0, 12),
                (0, 16),
                (4, 17),
            ),
        ),
        lambda name: CaseData(
            node_counts={
                agx.NodeType(type_id=0, num_connections=3): 4,
                agx.NodeType(type_id=1, num_connections=2): 6,
            },
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            graph_filename="rxx_4-3FG_6-2FG.json.gz",
            num_graphs=5,
            num_configs=0,
            name=name,
            doubles={0: True, 1: True, 2: True, 3: False, 5: False},
            parallels={0: True, 1: False, 2: True, 3: True, 5: False},
            iso_pass=(),
        ),
        lambda name: CaseData(
            node_counts={agx.NodeType(type_id=0, num_connections=4): 5},
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            graph_filename="rxx_5-4FG.json.gz",
            num_graphs=6,
            num_configs=0,
            name=name,
            doubles={0: True, 1: False, 2: True, 3: True, 4: True, 5: False},
            parallels={0: True, 1: True, 2: True, 3: True, 4: False, 5: True},
            iso_pass=(),
        ),
        lambda name: CaseData(
            node_counts={
                agx.NodeType(type_id=0, num_connections=3): 2,
                agx.NodeType(type_id=1, num_connections=2): 2,
                agx.NodeType(type_id=2, num_connections=1): 2,
            },
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            graph_filename="rxx_2-3FG_2-2FG_2-1FG.json.gz",
            num_graphs=2,
            num_configs=0,
            name=name,
            doubles={0: True, 2: False},
            parallels={0: False, 2: True},
            iso_pass=(),
        ),
        lambda name: CaseData(
            node_counts={
                agx.NodeType(type_id=0, num_connections=4): 3,
                agx.NodeType(type_id=1, num_connections=3): 3,
                agx.NodeType(type_id=2, num_connections=3): 1,
            },
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            graph_filename="rxx_3-4FG_4-3FG.json.gz",
            num_graphs=9,
            num_configs=4,
            name=name,
            doubles={
                0: True,
                1: True,
                2: True,
                3: True,
                4: True,
                5: True,
                6: False,
                7: True,
                8: False,
            },
            parallels={
                0: True,
                1: True,
                2: True,
                3: True,
                4: True,
                5: False,
                6: True,
                7: True,
                8: True,
            },
            iso_pass=(
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
                (8, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (8, 1),
                (4, 2),
                (6, 2),
                (0, 3),
                (3, 3),
                (4, 3),
            ),
        ),
    )
)
def graph_data(request: pytest.FixtureRequest) -> CaseData:
    """A fixture."""
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore[attr-defined]
    )


@pytest.fixture(params=["kamada", "spring", "spectral"])
def layout_type(request: pytest.FixtureRequest) -> CaseData:
    """A fixture."""
    return request.param


@pytest.fixture(params=[1, 5])
def scale(request: pytest.FixtureRequest) -> CaseData:
    """A fixture."""
    return request.param
