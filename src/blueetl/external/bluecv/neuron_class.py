"""neuron_class features from BlueCV."""

import logging
from functools import partial

import numpy as np
import pandas as pd
from elephant import spectral, statistics
from elephant.spike_train_correlation import correlation_coefficient
from quantities import ms

from blueetl.constants import GID, TIME
from blueetl.external.bluecv.utils import to_binned_spiketrain, to_spiketrains

L = logging.getLogger(__name__)


def calculate_features_by_neuron_class(repo, key, df, params):
    """Calculate features grouped by neuron_class."""
    t_start, t_stop = repo.windows.get_bounds(key.window)
    # create an array containing multiple arrays of spikes, one for each gid
    spiketrains = df.groupby([GID])[TIME].apply(np.array).to_numpy()
    ST = to_spiketrains(spiketrains, t_start, t_stop)
    BST = to_binned_spiketrain(ST)
    functions = {
        "PSD": partial(get_PSD, spiketrains),
        "AC": partial(get_AC, BST),
        "CPDF": partial(get_CPDF, ST),
        "PSTH": partial(get_PSTH, spiketrains, t_start=t_start, t_stop=t_stop),
    }
    result = {}
    for feature_name, feature_config in params.items():
        feature_params = feature_config.get("params", {})
        result[feature_name] = [functions[feature_name](**feature_params)]
    return {"by_neuron_class": pd.DataFrame(result)}


def get_PSD(spiketrains, n_segments=2):
    """Get power spectrum density of neuronal population."""
    _, PSD = spectral.welch_psd(
        np.concatenate(spiketrains),
        n_segments=n_segments,
    )
    return PSD


def get_AC(BST):
    """Get mean firing rate of neuronal population."""
    return np.triu(correlation_coefficient(BST), k=1)


def get_CPDF(ST, bin_size=20):
    """Get complexity."""
    CPDF = statistics.Complexity(spiketrains=ST, bin_size=bin_size * ms).pdf()
    return np.asarray(CPDF).reshape(-1)


def get_PSTH(spiketrains, t_start, t_stop, bin_size=20):
    """Get perstimulus time histogram of the selected population.

    Args:
        spiketrains: array containing multiple arrays of spikes
        t_start (float): start time
        t_stop (float): stop time
        bin_size (float): bin size in ms

    Returns:
        a list with two items:
            counts (np.ndarray): number of spike_times per bin in psth
            bins (np.ndarray): start of the bins
    """
    spiketrains = np.concatenate(spiketrains)
    inputbins = np.arange(t_start, t_stop, bin_size)
    counts, bins = np.histogram(
        spiketrains,
        bins=inputbins.size,
        range=(t_start, t_stop),
    )
    return [counts, bins[:-1]]
