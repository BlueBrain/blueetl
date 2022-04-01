import logging

import pandas as pd

from blueetl.constants import CIRCUIT_ID, COUNT, NEURON_CLASS
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class NeuronClasses(BaseExtractor):
    COLUMNS = [
        CIRCUIT_ID,
        NEURON_CLASS,
        COUNT,
    ]

    @classmethod
    def _validate(cls, df):
        cls._validate_columns(df, allow_extra=True)

    @classmethod
    def from_neurons(cls, neurons, target, neuron_classes, limit=None):
        """Load neuron classes information for each circuit."""
        results = []
        for index, count in neurons.count_by_neuron_class().etl.iter():
            # index: circuit_id, neuron_class
            neuron_class_conf = neuron_classes[index.neuron_class].copy()
            limit = neuron_class_conf.pop("$limit", limit)
            results.append(
                {
                    CIRCUIT_ID: index.circuit_id,
                    NEURON_CLASS: index.neuron_class,
                    COUNT: count,
                    "limit": limit,
                    "target": target,
                    **neuron_class_conf,
                }
            )
        df = pd.DataFrame(results)
        return cls(df)
