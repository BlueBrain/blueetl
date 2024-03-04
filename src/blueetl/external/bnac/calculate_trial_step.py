"""Trial steps functions adapted from BlueNetworkActivityComparison/bnac/onset.py."""

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

L = logging.getLogger(__name__)


def _get_bounds(params):
    lower_bound, upper_bound = params["bounds"]
    assert lower_bound <= 0
    assert upper_bound >= 0
    return lower_bound, upper_bound


def _histogram_from_spikes(spikes, params):
    lower_bound, upper_bound = _get_bounds(params)
    onset_test_window_length = upper_bound - lower_bound
    histogram, _ = np.histogram(
        spikes,
        range=(lower_bound, upper_bound),
        bins=int(onset_test_window_length * params["histo_bins_per_ms"]),
    )
    return histogram


def _onset_from_histogram(histogram, params):
    lower_bound, _ = _get_bounds(params)
    smoothed_histogram = gaussian_filter(histogram, sigma=params["smoothing_width"])

    pre_window_length = -lower_bound
    pre_window_bins = int(pre_window_length * params["histo_bins_per_ms"])
    pre_smoothed_histogram = smoothed_histogram[:pre_window_bins]
    post_smoothed_histogram = smoothed_histogram[pre_window_bins:]

    onset_dict = {
        "pre_mean": np.mean(pre_smoothed_histogram),
        "pre_std": np.std(pre_smoothed_histogram),
        "post_max": np.max(post_smoothed_histogram),
    }
    onset_dict["pre_mean_post_max_ratio"] = onset_dict["pre_mean"] / onset_dict["post_max"]

    threshold = onset_dict["pre_mean"] + params["threshold_std_multiple"] * onset_dict["pre_std"]
    where_above_thresh = np.where(post_smoothed_histogram > threshold)[0]
    index_above_std = 0
    if len(where_above_thresh) > 0:
        index_above_std = where_above_thresh[0]
    # index_above_count_thresh = np.where(post_smoothed_histogram > 15)[0][0]
    # index_above_count_thresh = -1000
    # cortical_onset_index = np.max([index_above_std, index_above_count_thresh])
    cortical_onset_index = index_above_std

    onset_dict["cortical_onset"] = (
        float(cortical_onset_index) / float(params["histo_bins_per_ms"]) + params["ms_post_offset"]
    )
    if params.get("figures_path"):
        _plot(smoothed_histogram, params, onset_dict)
    return onset_dict


def _plot(smoothed_histogram, params, onset_dict):
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    lower_bound, upper_bound = _get_bounds(params)
    plt.figure()
    sns.set()
    sns.set_style("ticks")
    x_vals = list(np.arange(lower_bound, upper_bound, 0.2))
    plt.plot(x_vals, smoothed_histogram)
    plt.scatter(
        onset_dict["cortical_onset"],
        [onset_dict["pre_mean"] + params["threshold_std_multiple"] * onset_dict["pre_std"]],
    )
    plt.gca().set_xlim([lower_bound, upper_bound])
    plt.gca().set_xlabel("Time (ms)")
    plt.gca().set_ylabel("Number of spikes")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    filepath = Path(params["base_path"], params["figures_path"], "plot.pdf")
    L.info("Figures path: %s", filepath)
    filepath.parent.mkdir(exist_ok=True)
    plt.savefig(filepath)
    plt.close()


def onset_from_spikes(spikes_list, params):
    """Calculate the cortical onset from a list of spikes, one for each trial.

    Args:
        spikes_list: list of spikes as numpy arrays.
        params: dictionary of parameters from the trial steps configuration.

    Returns:
        float representing the dynamic offset to be added to the initial offset of each trial step.
    """
    L.info(
        "onset_from_spikes: processing %s arrays of spikes using params=%r",
        len(spikes_list),
        params,
    )
    spikes = np.concatenate(spikes_list)
    histogram = _histogram_from_spikes(spikes, params)
    onset_dict = _onset_from_histogram(histogram, params)
    return onset_dict["cortical_onset"]
