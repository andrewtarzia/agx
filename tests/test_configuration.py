import pathlib

import pytest

import agx

from .case_data import CaseData


def test_configuration(graph_data: CaseData) -> None:
    """Test graph layout processes.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    config_directory = pathlib.Path(__file__).resolve().parent / "test_configs"
    config_directory.mkdir(exist_ok=True)

    iterator = agx.TopologyIterator(
        node_counts=graph_data.node_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )
    if graph_data.num_configs == 0:
        with pytest.raises(RuntimeError) as e_info:
            iterator.get_configurations()

        assert str(e_info.value) == (
            "modifiable_types is len 0. If 0, then you have no need to "
            "screen building block configurations. If greater than 2, then "
            "this code cannot handle this yet. Sorry!"
        )
    else:
        assert len(iterator.get_configurations()) == graph_data.num_configs

        run_topology_codes: list[agx.ConfiguredCode] = []
        # Check for iso checks, iterating over topology codes as well.
        for configured_code in iterator.yield_configured_codes():
            if agx.utilities.is_configured_code_isomoprhic(
                test_code=configured_code,
                run_topology_codes=run_topology_codes,
            ):
                assert (
                    configured_code.topology_code.idx,
                    configured_code.configuration.idx,
                ) in graph_data.iso_pass
                assert graph_data.iso_pass[len(run_topology_codes)] == (
                    configured_code.topology_code.idx,
                    configured_code.configuration.idx,
                )
                run_topology_codes.append(configured_code)

            bc_name = (
                config_directory / f"bc_{iterator.graph_type}_"
                f"{configured_code.topology_code.idx}_"
                f"{configured_code.configuration.idx}.txt"
            )

            if not bc_name.exists():
                with bc_name.open("w") as f:
                    f.write(
                        str(
                            configured_code.configuration.get_hashable_idx_dict()
                        )
                    )

            with bc_name.open("r") as f:
                lines = f.readlines()
            test = lines[0]
            assert (
                str(configured_code.configuration.get_hashable_idx_dict())
                == test
            )
