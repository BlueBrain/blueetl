# adapted from BlueNetworkActivityComparison/bnac/data_processor.py
import logging

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from blueetl.constants import BIN, COUNT, GID, NEURON_CLASS_INDEX, TIME, TIMES, TRIAL

L = logging.getLogger(__name__)
FIRST = "first"


def get_initial_spiking_stats_v1(repo, key, df, params):
    number_of_trials = repo.windows.get_number_of_trials(key.window)
    trial_columns = list(range(number_of_trials))
    duration = repo.windows.get_duration(key.window)
    neurons = repo.neurons.df.etl.q(circuit_id=key.circuit_id, neuron_class=key.neuron_class)
    # first spike for each trial and gid, averaged across all trials where the neuron was present
    first_spike_time_means_cort_zeroed = (
        df.groupby([TRIAL, GID]).min().groupby(GID).mean().reset_index()
    )
    # same, but including all the existing neurons in the given neuron class and using NaN
    first_spike_time_means_cort_zeroed_by_cell = pd.merge(
        neurons, first_spike_time_means_cort_zeroed, how="left"
    )

    # spike counts with columns [trial, gid, count]
    spike_counts_by_trial = (
        df.groupby([TRIAL, GID]).count().reset_index().rename(columns={TIME: COUNT})
    )
    # spike counts with index gid and columns [0, 1, 2...], one numeric column for each trial,
    # including a column even when there are no spikes in that trial. To include the empty columns,
    # it's not possible to use: df.pivot(index=GID, columns=TRIAL, values=COUNT)
    spike_counts_by_trial = pd.concat(
        [spike_counts_by_trial.etl.q(trial=i).set_index(GID)[COUNT] for i in trial_columns],
        axis=1,
        keys=trial_columns,
    )
    # spike counts array for all the neurons, using 0 for missing neurons
    spike_counts_by_trial_and_cell = (
        pd.merge(neurons, spike_counts_by_trial, how="left", on=GID)[trial_columns]
        .fillna(0)
        .to_numpy()
        .transpose()
    )
    mean_spike_counts = np.mean(spike_counts_by_trial_and_cell, axis=0)
    mean_of_spike_counts_for_each_trial = np.mean(spike_counts_by_trial_and_cell, axis=1)

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
        "all_spike_counts": spike_counts_by_trial_and_cell.flatten(),
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


def get_initial_spiking_stats_v2(repo, key, df, params):
    number_of_trials = repo.windows.get_number_of_trials(key.window)
    trial_columns = list(range(number_of_trials))
    duration = repo.windows.get_duration(key.window)

    # df with index (trial, gid) and columns (count, times)
    spikes_by_trial = df.groupby([TRIAL, GID])[TIME].agg(
        **{COUNT: "count", TIMES: list, FIRST: "min"}
    )
    # first spike for each trial and gid, averaged across all trials where the neuron was present
    first_spike_time_means_cort_zeroed = (
        spikes_by_trial[FIRST].groupby(GID).mean().rename("first_spike_time_means_cort_zeroed")
    )
    # add rows with NaN values
    spikes_by_trial = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [trial_columns, spikes_by_trial.etl.labels_of(GID)], names=[TRIAL, GID]
        ),
    ).join(spikes_by_trial)

    mean_spike_counts = spikes_by_trial[COUNT].fillna(0).groupby(GID).mean()
    mean_spike_counts = mean_spike_counts.rename("mean_spike_counts")
    mean_of_spike_counts_for_each_trial = (
        spikes_by_trial[COUNT]
        .fillna(0)
        .groupby(TRIAL)
        .mean()
        .rename("mean_of_spike_counts_for_each_trial")
    )

    mean_firing_rates_per_second = mean_spike_counts * 1000.0 / duration
    mean_firing_rates_per_second = mean_firing_rates_per_second.rename(
        "mean_firing_rates_per_second"
    )

    return {
        "spikes_by_trial": spikes_by_trial,
        "first_spike_time_means_cort_zeroed": first_spike_time_means_cort_zeroed,
        "mean_spike_counts": mean_spike_counts,
        "mean_firing_rates_per_second": mean_firing_rates_per_second,
        "mean_of_spike_counts_for_each_trial": mean_of_spike_counts_for_each_trial,
        # scalar values
        "mean_of_mean_spike_counts": np.nanmean(mean_spike_counts),
        "mean_of_mean_firing_rates_per_second": np.mean(mean_firing_rates_per_second),
        "std_of_mean_firing_rates_per_second": np.std(mean_firing_rates_per_second),
    }


def get_histogram_features(repo, key, df, params):
    number_of_trials = repo.windows.get_number_of_trials(key.window)
    duration = repo.windows.get_duration(key.window)
    t_start, t_stop = repo.windows.get_bounds(key.window)
    # all the spike times are concatenated regardless of the trial
    times = df[TIME].to_numpy()
    hist, _ = np.histogram(times, range=[t_start, t_stop], bins=int(duration))
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


def calculate_features_single(repo, key, df, params):
    spiking_stats = get_initial_spiking_stats_v1(repo, key, df, params)
    histogram_features = get_histogram_features(repo, key, df, params)
    return {**spiking_stats, **histogram_features}


def calculate_features_multi(repo, key, df, params):
    # all neurons, having spikes or not
    neurons = repo.neurons.df.etl.q(circuit_id=key.circuit_id, neuron_class=key.neuron_class)
    neurons = neurons.set_index(GID)[[NEURON_CLASS_INDEX]]
    number_of_trials = repo.windows.get_number_of_trials(key.window)
    trial_columns = list(range(number_of_trials))
    export_all_neurons = params.get("export_all_neurons", False)

    spiking_stats = get_initial_spiking_stats_v2(repo, key, df, params)
    histogram_features = get_histogram_features(repo, key, df, params)

    # df with (gid) as index, and features as columns
    if export_all_neurons:
        # return all the neurons, having spikes or not
        by_gid = neurons.join(
            [
                spiking_stats["first_spike_time_means_cort_zeroed"],
                spiking_stats["mean_spike_counts"],
                spiking_stats["mean_firing_rates_per_second"],
            ]
        )
    else:
        # return only neurons with spikes
        by_gid = pd.concat(
            [
                spiking_stats["first_spike_time_means_cort_zeroed"],
                spiking_stats["mean_spike_counts"],
                spiking_stats["mean_firing_rates_per_second"],
            ],
            axis=1,
        )

    # df with (trial, gid) as index, and features as columns
    if export_all_neurons:
        # return all the neurons, having spikes or not
        tmp_s = neurons[NEURON_CLASS_INDEX]
        tmp_s = pd.concat([tmp_s for _ in trial_columns], keys=trial_columns, names=[TRIAL])
        neurons_by_trial = pd.DataFrame(index=pd.MultiIndex.from_frame(tmp_s.reset_index()))
        neurons_by_trial = neurons_by_trial.reset_index(NEURON_CLASS_INDEX)
        by_gid_and_trial = neurons_by_trial.join(
            [
                spiking_stats["spikes_by_trial"],
            ]
        )
    else:
        # return only neurons with spikes
        by_gid_and_trial = pd.concat(
            [
                spiking_stats["spikes_by_trial"],
            ],
            axis=1,
        )

    # df with features as columns and a single row
    # the index will be dropped when concatenating because it's unnamed
    by_neuron_class = pd.DataFrame(
        {
            "mean_of_mean_spike_counts": spiking_stats["mean_of_mean_spike_counts"],
            "mean_of_mean_firing_rates_per_second": spiking_stats[
                "mean_of_mean_firing_rates_per_second"
            ],
            "std_of_mean_firing_rates_per_second": spiking_stats[
                "std_of_mean_firing_rates_per_second"
            ],
            "mean_of_spike_times_normalised_hist_1ms_bin": histogram_features[
                "mean_of_spike_times_normalised_hist_1ms_bin"
            ],
            "min_of_spike_times_normalised_hist_1ms_bin": histogram_features[
                "min_of_spike_times_normalised_hist_1ms_bin"
            ],
            "max_of_spike_times_normalised_hist_1ms_bin": histogram_features[
                "max_of_spike_times_normalised_hist_1ms_bin"
            ],
            "argmax_spike_times_hist_1ms_bin": histogram_features[
                "argmax_spike_times_hist_1ms_bin"
            ],
        },
        index=[0],
    )

    # df with (trial) as index, and features as columns
    by_neuron_class_and_trial = spiking_stats["mean_of_spike_counts_for_each_trial"].to_frame()

    # df with (bin) as index, and features as columns
    histograms = pd.DataFrame(
        {
            "spike_times_normalised_hist_1ms_bin": histogram_features[
                "spike_times_normalised_hist_1ms_bin"
            ],
            "spike_times_max_normalised_hist_1ms_bin": histogram_features[
                "spike_times_max_normalised_hist_1ms_bin"
            ],
            "smoothed_3ms_spike_times_max_normalised_hist_1ms_bin": histogram_features[
                "smoothed_3ms_spike_times_max_normalised_hist_1ms_bin"
            ],
        }
    ).rename_axis(BIN)

    return {
        "by_gid": by_gid,
        "by_gid_and_trial": by_gid_and_trial,
        "by_neuron_class": by_neuron_class,
        "by_neuron_class_and_trial": by_neuron_class_and_trial,
        "histograms": histograms,
    }
