import logging

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from blueetl.constants import GID, TIME, TRIAL, COUNT

L = logging.getLogger(__name__)


def get_initial_spiking_stats(repo, key, df, params):
    number_of_trials = repo.windows.get_number_of_trials(key.window)
    trial_columns = list(range(number_of_trials))
    duration = repo.windows.get_duration(key.window)
    neurons = repo.neurons.df.etl.q(circuit_id=key.circuit_id, neuron_class=key.neuron_class)
    # first spike for each trial and gid, averaged across all trials where the neuron was present
    first_spike_time_means_cort_zeroed = (
        df.groupby([TRIAL, GID]).first().groupby(GID).mean().reset_index()
    )
    # same, but including all the existing neurons in the given neuron class and using NaN
    first_spike_time_means_cort_zeroed_by_cell = pd.merge(
        neurons, first_spike_time_means_cort_zeroed, how="left"
    )

    # spike counts with columns [trial, gid, count]
    spike_counts_by_trial = (
        df.groupby([TRIAL, GID]).count().reset_index().rename(columns={TIME: COUNT})
    )
    # spike counts with index gid and columns [0, 1, 2...], one numeric column for each trial
    spike_counts_by_trial = spike_counts_by_trial.pivot(index=GID, columns=TRIAL, values=COUNT)
    # spike counts array for all the neurons, using 0 for missing neurons
    spike_counts_by_trial_by_cell = (
        pd.merge(neurons, spike_counts_by_trial, how="left", on=GID)[trial_columns]
        .fillna(0)
        .to_numpy()
        .transpose()
    )
    mean_spike_counts = np.mean(spike_counts_by_trial_by_cell, axis=0)
    mean_of_spike_counts_for_each_trial = np.mean(spike_counts_by_trial_by_cell, axis=1)

    mean_firing_rates_per_second = mean_spike_counts * 1000.0 / duration
    first_trial = df.etl.q(trial=0)[[GID, TIME]].sort_values([GID, TIME], ignore_index=True)
    spike_times_cort_zeroed_by_group_zeroed_gid = list(
        pd.merge(neurons, first_trial.groupby(GID).agg(list), how="left", on=GID)[TIME]
    )
    # both the arrays passed to searchsorted must be already sorted
    spike_group_zeroed_gids = np.searchsorted(neurons[GID], first_trial[GID])

    return {
        "first_spike_time_means_cort_zeroed_by_cell": (
            first_spike_time_means_cort_zeroed_by_cell[TIME].to_numpy()
        ),
        "first_spike_time_means_cort_zeroed": first_spike_time_means_cort_zeroed[TIME].to_numpy(),
        "all_spike_counts": spike_counts_by_trial_by_cell.flatten(),
        "mean_spike_counts": mean_spike_counts,
        "mean_of_mean_spike_counts": np.nanmean(mean_spike_counts),
        "non_zero_mean_spike_counts": mean_spike_counts[mean_spike_counts > 0],
        "mean_firing_rates_per_second": mean_firing_rates_per_second,
        "mean_of_mean_firing_rates_per_second": np.mean(mean_firing_rates_per_second),
        "std_of_mean_firing_rates_per_second": np.std(mean_firing_rates_per_second),
        "mean_of_spike_counts_for_each_trial": mean_of_spike_counts_for_each_trial,
        "spike_gids": first_trial[GID].to_numpy(),
        "spike_group_zeroed_gids": spike_group_zeroed_gids,
        "spike_times_cort_zeroed": first_trial[TIME].to_numpy(),
        "spike_times_cort_zeroed_by_group_zeroed_gid": spike_times_cort_zeroed_by_group_zeroed_gid,
    }


def get_histogram_features(repo, key, df, params):
    number_of_trials = repo.windows.get_number_of_trials(key.window)
    duration = repo.windows.get_duration(key.window)
    duration = int(duration)
    times = df[TIME].to_numpy()
    hist, _ = np.histogram(times, range=[0, duration], bins=duration)
    num_target_cells = len(
        repo.neurons.df.etl.q(circuit_id=key.circuit_id, neuron_class=key.neuron_class)
    )
    hist = hist / (num_target_cells * number_of_trials)
    min_hist = np.min(hist)
    max_hist = np.max(hist)
    norm_hist = hist / (max_hist or 1)
    smoothed_hist = gaussian_filter(hist, sigma=4.0)
    max_smoothed_hist = np.max(smoothed_hist)
    norm_smoothed_hist = smoothed_hist / (max_smoothed_hist or 1)
    return {
        "spike_times_normalised_hist_1ms_bin": hist,
        "mean_of_spike_times_normalised_hist_1ms_bin": np.mean(hist),
        "min_of_spike_times_normalised_hist_1ms_bin": min_hist,
        "max_of_spike_times_normalised_hist_1ms_bin": max_hist,
        "argmax_spike_times_hist_1ms_bin": np.argmax(hist),
        "spike_times_max_normalised_hist_1ms_bin": norm_hist,
        "smoothed_3ms_spike_times_max_normalised_hist_1ms_bin": norm_smoothed_hist,
    }
