"""Comparison features from BlueCV."""

import logging

from elephant.spike_train_dissimilarity import van_rossum_distance, victor_purpura_distance
from quantities import ms

L = logging.getLogger(__name__)


def get_VR(spiketrain1, spiketrain2, tau=10):
    """Get Van rossum distance between two spiketrains."""
    result = van_rossum_distance([spiketrain1, spiketrain2], tau * ms)[0, 1]
    # keep the real part because sometimes the result is complex?
    # example: SpikeTrain([4027.375, 4051.9, 4114.475], units="ms", t_stop=5000.0)
    return result.real


def get_VP(spiketrain1, spiketrain2, q=0.1):
    """Get Victor purpura distance between two spiketrains."""
    return victor_purpura_distance([spiketrain1, spiketrain2], q * (1 / ms))[0, 1]
