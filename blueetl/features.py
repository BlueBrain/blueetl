import numpy as np
from elephant import spectral, statistics
from elephant.spike_train_correlation import correlation_coefficient
from elephant.spike_train_dissimilarity import van_rossum_distance, victor_purpura_distance
from quantities import ms

# gid features


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
    """Get measure of local variation for a sequence of time intervals between events of a neuron"""
    return statistics.lv(statistics.isi(spiketrain), with_nan=True)


def get_latency(spiketrain, t_start):
    """Get latency of a single neuron (time until first spike)."""
    return spiketrain[0] - t_start


def get_spike_count(spiketrain):
    """Get number of spikes in a time window per gid."""
    return spiketrain.size


# neuron class features


def get_PSD(spiketrains, n_segments=2):
    """Get power spectrum density of neuronal population"""
    _, PSD = spectral.welch_psd(
        np.concatenate(spiketrains),
        n_segments=n_segments,
    )
    return PSD


def get_AC(BST):
    """Get mean firing rate of neuronal population"""
    return np.triu(correlation_coefficient(BST), k=1)


def get_CPDF(ST, bin_size=20):
    """Get complexity"""
    CPDF = statistics.complexity_pdf(ST, bin_size=bin_size * ms)
    return np.asarray(CPDF).reshape(-1)


def get_PSTH(spiketrains, t_start, t_stop, bin_size=20):
    """Get perstimulus time histogram of the selected population.

    Args:
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


# comparison features


def get_VR(spiketrain1, spiketrain2, tau=10):
    """Get Van rossum distance between two spiketrains."""
    result = van_rossum_distance([spiketrain1, spiketrain2], tau * ms)[0, 1]
    # keep the real part because sometimes the result is complex?
    # example: SpikeTrain([4027.375, 4051.9, 4114.475], units="ms", t_stop=5000.0)
    return result.real


def get_VP(spiketrain1, spiketrain2, q=0.1):
    """Get Victor purpura distance between two spiketrains."""
    return victor_purpura_distance([spiketrain1, spiketrain2], q * (1 / ms))[0, 1]
