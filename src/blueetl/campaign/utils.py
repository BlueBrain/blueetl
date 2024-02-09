"""Simulation campaign utils.

Originally based on:
https://bbpgitlab.epfl.ch/nse/bbp-workflow/-/blob/2032ffd6/bbp_workflow/simulation/util.py
"""

from collections.abc import Iterator
from typing import Any, Optional, Union

import xarray as xr

from blueetl.campaign.config import SimulationCampaign

DataArrayOrDict = Union[dict, xr.DataArray]


def from_xarray(config: DataArrayOrDict) -> SimulationCampaign:
    """Return a SimulationCampaign from xarray.DataArray or dict."""
    if isinstance(config, dict):
        return SimulationCampaign.from_xarray_dict(config)
    else:
        return SimulationCampaign.from_xarray(config)


def campaign_sims(
    config: DataArrayOrDict, include_empty: bool = False
) -> Iterator[tuple[dict[str, Any], Optional[str]]]:
    """Iterate simulations from the campaign.

    Campaign sim can be empty if it was filtered out by `coords_filter_func`.

    Args:
        config (xarray.DataArray|dict): Simulation campaign configuration.
        include_empty (bool): If true, all sims are included(even empty ones).

    Yields:
        tuple(dict, str): Simulation conditions and full path to individual simulation.
                          Path can be `None` if sim folder is empty.
    """
    config = from_xarray(config)
    for sim in config:
        if include_empty or not sim.empty:
            yield sim.conditions, sim.path or None


def _campaign_sims_with_indices(config: DataArrayOrDict) -> Iterator[tuple[dict, dict, str]]:
    """Iterate all simulations from the non-coupled coords campaign with indices.

    Args:
        config (xarray.DataArray|dict): Simulation campaign configuration.

    Yields:
        tuple(dict, dict, str): Simulation conditions with locs/indices and full path to
                                individual simulations.
    """
    config = from_xarray(config)
    if config.is_coupled():
        raise ValueError("Please provide non-coupled coords sim campaign!")
    # number of distinct values for each condition
    totals = config.conditions.nunique().to_dict()
    # indices for each condition and simulation
    idxs = {cond: series.cat.codes for cond, series in config.conditions.astype("category").items()}
    for sim in config:
        indices = {"idx": sim.index}
        for cond in sim.conditions:
            indices[f"{cond}_idx"] = idxs[cond].at[sim.index]
            indices[f"{cond}_total"] = totals[cond]
        yield sim.conditions, indices, sim.path


def _campaign_sim_indices(config: DataArrayOrDict) -> list[int]:
    """Non-empty simulation indices of the campaign."""
    config = from_xarray(config)
    return [sim.index for sim in config if sim.path]


def _campaign_sim_index_to_coords(config: DataArrayOrDict) -> dict[int, dict[str, Any]]:
    """Index to coords dict of the campaign sims."""
    config = from_xarray(config)
    return {sim.index: sim.conditions for sim in config if sim.path}
