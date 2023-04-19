"""NeuronClasses extractor."""
import json
import logging
from typing import Optional

import pandas as pd

from blueetl.config.analysis_model import NeuronClassConfig
from blueetl.constants import CIRCUIT_ID, COUNT, GIDS, LIMIT, NEURON_CLASS, QUERY, TARGET
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
        TARGET,
        GIDS,
        QUERY,
    ]

    @classmethod
    def from_neurons(
        cls,
        neurons: Neurons,
        target: Optional[str],
        neuron_classes: dict[str, NeuronClassConfig],
        limit: Optional[int] = None,
    ) -> "NeuronClasses":
        """Load neuron classes information for each circuit.

        Args:
            neurons: Neurons extractor.
            target: target string, or None to not filter by target.
            neuron_classes: configuration dict of neuron classes to be extracted.
            limit: if specified, limit the number of extracted neurons.

        Returns:
            NeuronClasses: new instance.
        """
        results = []
        for index, count in neurons.count_by_neuron_class().etl.iter():
            # index: circuit_id, neuron_class
            config = neuron_classes[index.neuron_class]
            neuron_limit = limit if config.limit is None else config.limit
            neuron_target = target if config.target is None else config.target
            # gids and query are saved for reference and inspection.
            # query is saved as json string to avoid this error during the conversion to parquet:
            # pyarrow.lib.ArrowInvalid: 'cannot mix list and non-list, non-null values'
            results.append(
                {
                    CIRCUIT_ID: index.circuit_id,
                    NEURON_CLASS: index.neuron_class,
                    COUNT: count,
                    LIMIT: neuron_limit,
                    TARGET: neuron_target,
                    GIDS: json.dumps(config.gids) if config.gids else None,
                    QUERY: json.dumps(config.query),
                }
            )
        df = pd.DataFrame(results)
        return cls(df, cached=False, filtered=False)
