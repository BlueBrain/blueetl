import logging

import numpy as np
import pandas as pd
from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
from quantities import ms
from scipy.ndimage import gaussian_filter

from blueetl import etl, features
from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import CIRCUIT_ID, GID, NEURON_CLASS, TIME, WINDOW
from blueetl.repository import Repository
from blueetl.utils import load_yaml

L = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, analysis_config, use_cache=False):
        self.analysis_config = analysis_config
        self.repo = Repository(
            simulations_config=SimulationsConfig.load(analysis_config["simulation_campaign"]),
            extraction_config=analysis_config["extraction"],
            cache_dir=self.analysis_config["output"],
            use_cache=use_cache,
        )

    def initialize(self):
        self.repo.extract()
        self.repo.print()

    def _get_window_limits(self, window):
        return self.analysis_config["extraction"]["windows"][window]

    def calculate_features(self):
        records = []
        for key, df in self.repo.spikes.grouped_by_neuron_class():
            record = key._asdict()
            record.update(
                self._get_initial_spiking_stats(
                    circuit_id=record[CIRCUIT_ID],
                    neuron_class=record[NEURON_CLASS],
                    window=record[WINDOW],
                    df=df,
                )
            )
            record.update(
                self._get_histogram_features(
                    circuit_id=record[CIRCUIT_ID],
                    neuron_class=record[NEURON_CLASS],
                    window=record[WINDOW],
                    times=df[TIME],
                )
            )
            records.append(record)
        # in the returned df, the type of `neuron_class` and `window` is `object`
        return pd.DataFrame(records)

    def _get_initial_spiking_stats(self, circuit_id, neuron_class, window, df):
        # assuming number_of_trials = 1
        neurons = self.repo.neurons.df.etl.query_params(
            circuit_id=circuit_id, neuron_class=neuron_class
        )
        # first spike times
        first_spikes = df.groupby(GID).min().reset_index()
        # first spike times using NaN for missing neurons
        first_spikes_all = pd.merge(neurons, first_spikes, how="left")[TIME].to_numpy()
        # spike counts
        spike_counts = df.groupby(GID).count().reset_index()
        # spike counts using 0 for missing neurons
        spike_counts_all = pd.merge(neurons, spike_counts, how="left")[TIME].fillna(0).to_numpy()

        t_start, t_stop = self._get_window_limits(window)
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

    def _get_histogram_features(self, circuit_id, neuron_class, window, times):
        number_of_trials = self.analysis_config.get("number_of_trials", 1)
        t_start, t_stop = self._get_window_limits(window)
        hist, _ = np.histogram(times, range=[t_start, t_stop], bins=t_stop - t_start)
        num_target_cells = len(
            self.repo.neurons.df.etl.query_params(
                circuit_id=circuit_id,
                neuron_class=neuron_class,
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

    def calculate_features_by_gid(self):
        records = []
        for key, df in self.repo.spikes.grouped_by_gid():
            record = key._asdict()
            record.update(
                self._get_bluecv_features_by_gid(
                    circuit_id=record[CIRCUIT_ID],
                    neuron_class=record[NEURON_CLASS],
                    window=record[WINDOW],
                    gid=record[GID],
                    df=df,
                )
            )
            records.append(record)
        # in the returned df, the type of `neuron_class` and `window` is `object`
        return pd.DataFrame(records)

    def _get_bluecv_features_by_gid(self, circuit_id, neuron_class, window, gid, df):
        t_start, t_stop = self._get_window_limits(window)
        spiketrain = df[TIME].to_numpy()
        return {
            "MFR": features.get_MFR(spiketrain, t_start=t_start, t_stop=t_stop),
            "ISI": features.get_ISI(spiketrain),
            "CV": features.get_CV(spiketrain),
            "LV": features.get_LV(spiketrain),
            "latency": features.get_latency(spiketrain, t_start=t_start),
            "spike_count": features.get_spike_count(spiketrain),
        }

    def calculate_features_by_neuron_class(self):
        records = []
        for key, df in self.repo.spikes.grouped_by_neuron_class():
            record = key._asdict()
            record.update(
                self._get_bluecv_features_by_neuron_class(
                    circuit_id=record[CIRCUIT_ID],
                    neuron_class=record[NEURON_CLASS],
                    window=record[WINDOW],
                    df=df,
                )
            )
            records.append(record)
        # in the returned df, the type of `neuron_class` and `window` is `object`
        return pd.DataFrame(records)

    def _get_bluecv_features_by_neuron_class(self, circuit_id, neuron_class, window, df):
        t_start, t_stop = self._get_window_limits(window)
        # create an array containing multiple arrays of spikes, one for each gid
        spiketrains = df.groupby([GID])[TIME].apply(np.array).to_numpy()
        ST = to_spiketrains(spiketrains, t_start, t_stop)
        BST = to_binned_spiketrain(ST)
        return {
            "PSD": features.get_PSD(spiketrains, n_segments=2),
            "AC": features.get_AC(BST),
            "CPDF": features.get_CPDF(ST, bin_size=20),
            "PSTH": features.get_PSTH(spiketrains, t_start=t_start, t_stop=t_stop, bin_size=10),
        }


def to_spiketrains(data, t_start, t_end):
    return [SpikeTrain(spiketrain * ms, t_start=t_start, t_stop=t_end) for spiketrain in data]


def to_binned_spiketrain(ST, bin_size=5 * ms):
    return BinnedSpikeTrain(ST, bin_size=bin_size)


def main():
    loglevel = logging.INFO
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=logformat, level=loglevel)
    np.random.seed(0)
    etl.register_accessors()
    # analysis_config = load_yaml("./tests/data/tmp/analysis_config_01.yaml")
    analysis_config = load_yaml("./tests/data/tmp/analysis_config_02.yaml")
    a = Analyzer(analysis_config, use_cache=True)
    a.initialize()
    features = a.calculate_features()
    print("### features")
    print(features)


if __name__ == "__main__":
    main()
