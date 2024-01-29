"""Trial steps functions adapted from BlueNetworkActivityComparison/bnac/onset.py."""

import numpy as np
from scipy.ndimage import gaussian_filter


def _histogram_from_spikes(spikes, params):
    onset_test_window_length = params["post_window"][1] - params["pre_window"][0]
    histogram, _ = np.histogram(
        spikes,
        range=[params["pre_window"][0], params["post_window"][1]],
        bins=int(onset_test_window_length * params["histo_bins_per_ms"]),
    )
    return histogram


def _onset_from_histogram(histogram, params):
    onset_dict = {}
    smoothed_histogram = gaussian_filter(histogram, sigma=params["smoothing_width"])

    onset_pre_window_length = params["pre_window"][1] - params["pre_window"][0]
    onset_zeroed_post_start = params["post_window"][0] - params["pre_window"][0]
    pre_smoothed_histogram = smoothed_histogram[
        : onset_pre_window_length * params["histo_bins_per_ms"]
    ]
    post_smoothed_histogram = smoothed_histogram[
        onset_zeroed_post_start * params["histo_bins_per_ms"] :
    ]

    onset_dict["pre_mean"] = np.mean(pre_smoothed_histogram)
    onset_dict["pre_std"] = np.std(pre_smoothed_histogram)
    onset_dict["post_max"] = np.max(post_smoothed_histogram)
    onset_dict["pre_mean_post_max_ratio"] = onset_dict["pre_mean"] / onset_dict["post_max"]

    where_above_thresh = np.where(
        post_smoothed_histogram
        > (onset_dict["pre_mean"] + params["threshold_std_multiple"] * onset_dict["pre_std"])
    )[0]
    index_above_std = 0
    if len(where_above_thresh) > 0:
        index_above_std = where_above_thresh[0]
    # index_above_count_thresh = np.where(post_smoothed_histogram > 15)[0][0]
    # index_above_count_thresh = -1000
    # cortical_onset_index = np.max([index_above_std, index_above_count_thresh])
    cortical_onset_index = index_above_std

    onset_dict["cortical_onset"] = (
        float(cortical_onset_index) / float(params["histo_bins_per_ms"])
        + float(params["post_window"][0])
        + params["ms_post_offset"]
    )
    if params["fig_paths"]:
        _plot(smoothed_histogram, params, onset_dict)
    onset_dict["trial_steps_value"] = onset_dict["cortical_onset"]
    return onset_dict


def _plot(smoothed_histogram, params, onset_dict):
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    sns.set()
    sns.set_style("ticks")
    x_vals = list(np.arange(params["pre_window"][0], params["post_window"][1], 0.2))
    plt.plot(x_vals, smoothed_histogram)
    plt.scatter(
        onset_dict["cortical_onset"],
        [onset_dict["pre_mean"] + params["threshold_std_multiple"] * onset_dict["pre_std"]],
    )
    plt.gca().set_xlim([params["pre_window"][0], params["post_window"][1]])
    plt.gca().set_xlabel("Time (ms)")
    plt.gca().set_ylabel("Number of spikes")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    for fig_path in params["fig_paths"]:
        plt.savefig(fig_path)
    plt.close()


def onset_from_spikes(spikes, params):
    """Calculate trial steps from spikes."""
    histogram = _histogram_from_spikes(spikes, params)
    onset_dict = _onset_from_histogram(histogram, params)
    return onset_dict
