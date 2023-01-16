"""Calculate features for soma reports."""
import pandas as pd


def calculate_features_by_simulation_circuit_neuron_class_window(repo, key, df, params):
    """Calculate features for soma reports.

    The passed df must be grouped by simulation_id, circuit_id, neuron_class, window.
    """
    # pylint: disable=unused-argument
    assert key._fields == ("simulation_id", "circuit_id", "neuron_class", "window")
    by_neuron_class = pd.DataFrame(
        {
            "mean": df["value"].mean(),
            "std": df["value"].std(),
        },
        index=[0],
    )
    return {
        "by_neuron_class_v1": by_neuron_class,
    }


def calculate_features_by_simulation_circuit(repo, key, df, params):
    """Calculate features for soma reports.

    The passed df must be grouped by simulation_id, circuit_id.
    """
    # pylint: disable=unused-argument
    assert key._fields == ("simulation_id", "circuit_id")
    groupby = ["neuron_class", "window"]
    by_neuron_class = df.groupby(groupby)["value"].agg(["mean", "std"]).reset_index()
    return {
        "by_neuron_class_v2": by_neuron_class,
    }
