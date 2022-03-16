from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
from quantities import ms


def to_spiketrains(arrays, t_start, t_stop):
    return [SpikeTrain(spiketrain * ms, t_start=t_start, t_stop=t_stop) for spiketrain in arrays]


def to_binned_spiketrain(ST, bin_size=5 * ms):
    return BinnedSpikeTrain(ST, bin_size=bin_size)
