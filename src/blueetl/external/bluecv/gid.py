"""gid features from BlueCV."""

import logging
from functools import partial

import numpy as np
from elephant import statistics

from blueetl.constants import TIME

L = logging.getLogger(__name__)


def calculate_features_by_gid(repo, key, df, params):
    """Calculate features grouped by gid."""
    t_start, t_stop = repo.windows.get_bounds(key.window)
    spiketrain = df[TIME].to_numpy()
    functions = {
        "MFR": partial(get_MFR, spiketrain, t_start=t_start, t_stop=t_stop),
        "ISI": partial(get_ISI, spiketrain),
        "CV": partial(get_CV, spiketrain),
        "LV": partial(get_LV, spiketrain),
        "latency": partial(get_latency, spiketrain, t_start=t_start),
        "spike_count": partial(get_spike_count, spiketrain),
    }
    result = {}
    for feature_name, feature_config in params.items():
        feature_params = feature_config.get("params", {})
        result[feature_name] = functions[feature_name](**feature_params)
    return result


def get_MFR(spiketrain, t_start, t_stop):
    """Get mean firing rate of a neuron."""
    return statistics.mean_firing_rate(spiketrain, t_start=t_start, t_stop=t_stop)


def get_ISI(spiketrain):
    """Get interspike intervals of a neuron."""
    isi = statistics.isi(spiketrain)
    return isi[0] if isi.size > 0 else np.nan


def get_CV(spiketrain):
    """Get coefficient of variation of a neuron."""
    return statistics.cv(statistics.isi(spiketrain))


def get_LV(spiketrain):
    """Get measure of local variation for sequence of time intervals between events of a neuron."""
    return statistics.lv(statistics.isi(spiketrain), with_nan=True)


def get_latency(spiketrain, t_start):
    """Get latency of a single neuron (time until first spike)."""
    return spiketrain[0] - t_start


def get_spike_count(spiketrain):
    """Get number of spikes in a time window per gid."""
    return spiketrain.size
