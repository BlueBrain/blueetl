"""Common utils from BlueCV."""

from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
from quantities import ms


def to_spiketrains(arrays, t_start, t_stop):
    """Convert arrays to spiketrains."""
    return [SpikeTrain(spiketrain * ms, t_start=t_start, t_stop=t_stop) for spiketrain in arrays]


def to_binned_spiketrain(ST, bin_size=5 * ms):
    """Convert a spiketrain to a binned spiketrain."""
    return BinnedSpikeTrain(ST, bin_size=bin_size)
