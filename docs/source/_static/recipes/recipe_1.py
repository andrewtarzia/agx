"""Copiable code from Recipe #1."""  # noqa: INP001

import logging
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import agx

logger = logging.getLogger(__name__)


def main() -> None:
    """Run script."""
    # Define working directories.
    wd = (
        pathlib.Path(__file__).resolve().parent
        / ".."
        / ".."
        / "recipes"
        / "recipe_1_output"
    )
    wd.mkdir(exist_ok=True)

    node_counts = [
        {
            agx.NodeType(type_id=0, num_connections=4): 2,
            agx.NodeType(type_id=1, num_connections=2): 4,
        },
        {
            agx.NodeType(type_id=0, num_connections=4): 3,
            agx.NodeType(type_id=1, num_connections=2): 3,
            agx.NodeType(type_id=2, num_connections=2): 3,
        },
        {
            agx.NodeType(type_id=0, num_connections=3): 4,
            agx.NodeType(type_id=1, num_connections=2): 6,
        },
        {agx.NodeType(type_id=0, num_connections=4): 5},
        {
            agx.NodeType(type_id=0, num_connections=3): 2,
            agx.NodeType(type_id=1, num_connections=2): 2,
            agx.NodeType(type_id=2, num_connections=1): 2,
        },
    ]

    datas: dict[str | None, dict[int, dict[str, float]]] = {}
    for nc in node_counts:
        iterator = agx.TopologyIterator(node_counts=nc)
        datas[iterator.graph_type] = {}  # type: ignore[index]
        for tc in iterator.yield_graphs():
            nx_graph = tc.get_nx_graph()
            rd = nx.resistance_distance(nx_graph)
            mean_resistance_distance = np.mean(
                [j for i in rd for j in rd[i].values() if j != 0.0]
            )
            eccentricity_mean = np.mean(
                list(nx.eccentricity(nx_graph).values())
            )
            datas[iterator.graph_type][tc.idx] = {  # type: ignore[index]
                "eccentricity_mean": eccentricity_mean,
                "mean_resistance_distance": mean_resistance_distance,
            }
            layout = tc.get_layout(layout_type="kamada", scale=10)  # noqa: F841
            # Do something...

    fig, ax = plt.subplots(figsize=(5, 5))
    for ntype, ndata in datas.items():
        xs = [i["eccentricity_mean"] for i in ndata.values()]
        ys = [i["mean_resistance_distance"] for i in ndata.values()]

        ax.scatter(
            xs,
            ys,
            edgecolor="k",
            s=60,
            marker="o",
            alpha=1.0,
            label=ntype,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("mean eccentricity", fontsize=16)
    ax.set_ylabel("mean resistance distance", fontsize=16)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(wd / "graph_properties.png", dpi=360, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
