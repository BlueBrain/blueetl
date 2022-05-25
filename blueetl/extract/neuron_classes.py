import logging
from typing import Any, Dict, Optional

import pandas as pd

from blueetl.constants import CIRCUIT_ID, COUNT, NEURON_CLASS
from blueetl.extract.base import BaseExtractor
from blueetl.extract.neurons import Neurons

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
    def from_neurons(
        cls,
        neurons: Neurons,
        target: str,
        neuron_classes: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> "NeuronClasses":
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
        if not results:
            raise RuntimeError("All neuron classes are empty")
        df = pd.DataFrame(results)
        return cls(df)
