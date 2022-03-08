import logging

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from blueetl.constants import GID, TIME

L = logging.getLogger(__name__)


def get_initial_spiking_stats(analysis, key, df, params):
    # assuming number_of_trials = 1
    neurons = analysis.repo.neurons.df.etl.query_params(
        circuit_id=key.circuit_id, neuron_class=key.neuron_class
    )
    # first spike times
    first_spikes = df.groupby(GID).min().reset_index()
    # first spike times using NaN for missing neurons
    first_spikes_all = pd.merge(neurons, first_spikes, how="left")[TIME].to_numpy()
    # spike counts
    spike_counts = df.groupby(GID).count().reset_index()
    # spike counts using 0 for missing neurons
    spike_counts_all = pd.merge(neurons, spike_counts, how="left")[TIME].fillna(0).to_numpy()

    t_start, t_stop = analysis.get_window_limits(key.window)
    mean_firing_rates_per_second = spike_counts_all * 1000.0 / (t_stop - t_start)
    return {
        "first_spike_time_means_cort_zeroed_by_cell": first_spikes_all,
        "first_spike_time_means_cort_zeroed": first_spikes,
        "all_spike_counts": spike_counts_all,
        "mean_spike_counts": spike_counts_all,  # number_of_trials = 1
        "mean_of_mean_spike_counts": np.mean(spike_counts_all),  # number_of_trials = 1
        "non_zero_mean_spike_counts": spike_counts,  # number_of_trials = 1
        "mean_firing_rates_per_second": mean_firing_rates_per_second,
        "mean_of_mean_firing_rates_per_second": np.mean(mean_firing_rates_per_second),
        "std_of_mean_firing_rates_per_second": np.std(mean_firing_rates_per_second),
        # "mean_of_spike_counts_for_each_trial": np.mean(spike_counts_all, axis=1),
        # "spike_gids": df[GID].to_numpy(),
        # "spike_group_zeroed_gids": df[GID].to_numpy() - df[GID].min()),
        # "spike_times_cort_zeroed": [],
        # "spike_times_cort_zeroed_by_group_zeroed_gid": [],
    }


def get_histogram_features(analysis, key, df, params):
    number_of_trials = analysis.analysis_config.get("number_of_trials", 1)
    times = df[TIME]
    t_start, t_stop = analysis.get_window_limits(key.window)
    hist, _ = np.histogram(times, range=[t_start, t_stop], bins=t_stop - t_start)
    num_target_cells = len(
        analysis.repo.neurons.df.etl.query_params(
            circuit_id=key.circuit_id,
            neuron_class=key.neuron_class,
        )
    )
    hist = hist / (num_target_cells * number_of_trials)
    min_hist = np.min(hist)
    max_hist = np.max(hist)
    norm_hist = hist / (max_hist or 1)
    smoothed_hist = gaussian_filter(hist, sigma=4.0)
    max_smoothed_hist = np.max(smoothed_hist)
    norm_smoothed_hist = smoothed_hist / (max_smoothed_hist or 1)
    return {
        # hist_1ms_centers_cort_zeroed? [array of times of the interval with 1ms step]
        "spike_times_normalised_hist_1ms_bin": hist,
        "mean_of_spike_times_normalised_hist_1ms_bin": np.mean(hist),
        "min_of_spike_times_normalised_hist_1ms_bin": min_hist,
        "max_of_spike_times_normalised_hist_1ms_bin": max_hist,
        "argmax_spike_times_hist_1ms_bin": np.argmax(hist),
        "spike_times_max_normalised_hist_1ms_bin": norm_hist,
        "smoothed_3ms_spike_times_max_normalised_hist_1ms_bin": norm_smoothed_hist,
    }
