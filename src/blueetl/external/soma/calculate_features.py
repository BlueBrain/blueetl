"""Calculate features for soma reports."""


def calculate_features_by_simulation_circuit(repo, key, df, params):
    """Calculate features for soma reports.

    The passed df must be grouped by simulation_id, circuit_id.
    """
    # pylint: disable=unused-argument
    assert key._fields == ("simulation_id", "circuit_id")
    groupby = ["neuron_class", "window"]
    by_neuron_class = df.groupby(groupby, observed=True)["value"].agg(["mean", "std"]).reset_index()
    return {
        "by_neuron_class": by_neuron_class,
    }
