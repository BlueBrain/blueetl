"""NeuronClasses extractor."""

import json
import logging

import pandas as pd

from blueetl.config.analysis_model import NeuronClassConfig
from blueetl.constants import (
    CIRCUIT_ID,
    COUNT,
    GIDS,
    LIMIT,
    NEURON_CLASS,
    NODE_SET,
    POPULATION,
    QUERY,
)
from blueetl.extract.base import BaseExtractor
from blueetl.extract.neurons import Neurons

L = logging.getLogger(__name__)


class NeuronClasses(BaseExtractor):
    """NeuronClasses extractor class."""

    COLUMNS = [
        CIRCUIT_ID,
        NEURON_CLASS,
        COUNT,
        LIMIT,
        POPULATION,
        NODE_SET,
        GIDS,
        QUERY,
    ]

    @classmethod
    def from_neurons(
        cls, neurons: Neurons, neuron_classes: dict[str, NeuronClassConfig]
    ) -> "NeuronClasses":
        """Load neuron classes information for each circuit.

        Args:
            neurons: Neurons extractor.
            neuron_classes: configuration dict of neuron classes to be extracted.

        Returns:
            NeuronClasses: new instance.
        """
        results = []
        for index, count in neurons.count_by_neuron_class().etl.iter():
            # index: circuit_id, neuron_class
            config = neuron_classes[index.neuron_class]
            # gids and query are saved for reference and inspection.
            # query is saved as json string to avoid this error during the conversion to parquet:
            # pyarrow.lib.ArrowInvalid: 'cannot mix list and non-list, non-null values'
            results.append(
                {
                    CIRCUIT_ID: index.circuit_id,
                    NEURON_CLASS: index.neuron_class,
                    COUNT: count,
                    LIMIT: config.limit,
                    POPULATION: config.population,
                    NODE_SET: config.node_set,
                    GIDS: json.dumps(config.node_id) if config.node_id is not None else None,
                    QUERY: json.dumps(config.query),
                }
            )
        df = pd.DataFrame(results)
        return cls(df, cached=False, filtered=False)
