#!/usr/bin/env python
import hashlib
import logging
import os
import sys

import bluepy
import numpy as np
import pandas
import numpy
import tqdm

from blueetl import etl

circuit_rel_entries = ['cells', 'morphologies', 'emodels', 'connectome', 'atlas', 'projections']


def parse_arguments():
    import sys
    import json
    args = sys.argv[1:]
    if len(args) < 2:
        print("""Usage:
        {0} simulations.pkl config_file.json
        For details, see included README.txt
        """.format(__file__))
        sys.exit(2)
    with open(args[1], "r") as fid:
        cfg = json.load(fid)
    sims = pandas.read_pickle(args[0])
    return sims, cfg


def initial_setup(sims):
    def _calculate_hash(config):
        circ_cfg = {k: config.get(k, "__NONE__") for k in circuit_rel_entries}
        return hashlib.md5(str(circ_cfg).encode("UTF-8")).hexdigest()

    def _load_sim_and_circuit(path):
        simulation = bluepy.Simulation(path)
        circuit_hash = _calculate_hash(simulation.circuit.config)
        return simulation, circuit_hash

    def _func(x):
        for index, path in x.etl.iter_named_items():
            simulation, circuit_hash = _load_sim_and_circuit(path)
            yield simulation, {**index._asdict(), "circuit_hash": circuit_hash}

    data = sims.etl.remodel(_func)
    circuits = {index.circuit_hash: sim.circuit for index, sim in data.etl.iter_named_items()}
    return data, circuits


def read_spikes(sim):
    raw_spikes = sim.spikes.get()
    return raw_spikes


def get_sim_gids(sim):
    return sim.target_gids


def instant_target(circ, properties, **kwargs):
    c = circ.cells.get(properties=properties)
    c["gid"] = c.index
    for k, n in kwargs.items():
        c[k] = numpy.digitize(c[k], bins=numpy.linspace(c[k].min(), c[k].max() + 1E-12, n))
    return c.groupby(properties).agg(list)["gid"]


def flatten_locations(locations, flatmap):
    if isinstance(flatmap, list):
        flat_locations = locations[flatmap].values
    else:
        from voxcell import VoxelData
        fm = VoxelData.load_nrrd(flatmap)
        flat_locations = fm.lookup(locations.values).astype(float)
        flat_locations[flat_locations == -1] = numpy.NaN
    return pandas.DataFrame(flat_locations, index=locations.index)


def make_bins(flat_locations, nbins=1000):
    mn = numpy.nanmin(flat_locations, axis=0)
    mx = numpy.nanmax(flat_locations, axis=0)
    ratio = (mx[1] - mn[1]) / (mx[0] - mn[0]) # ratio * nx ** 2 = nbins
    nx = int(numpy.sqrt(nbins / ratio))
    ny = int(nbins / nx)
    binsx = numpy.linspace(mn[0], mx[0] + 1E-3, nx + 1)
    binsy = numpy.linspace(mn[1], mx[1] + 1E-3, ny + 1)
    return binsx, binsy


def make_t_bins(sims, t_start=None, t_end=None, t_step=20.0):
    if t_start is None:
        t_start = numpy.min(sims.map(lambda sim: sim.t_start))
    if t_end is None:
        t_end = numpy.max(sims.map(lambda sim: sim.t_end))
    t_bins = numpy.arange(t_start, t_end + t_step, t_step)
    return t_bins


def make_histogram_function(t_bins, loc_bins, location_dframe):
    t_step = numpy.mean(numpy.diff(t_bins))
    fac = 1000.0 / t_step
    nrns_per_bin = numpy.histogram2d(location_dframe.values[:, 0],
                                     location_dframe.values[:, 1],
                                     bins=loc_bins)[0]
    nrns_per_bin = nrns_per_bin.reshape((1,) + nrns_per_bin.shape)

    def spiking3dhistogram(spikes):
        t = spikes.index.values
        loc = location_dframe.loc[spikes].values
        raw, _ = numpy.histogramdd((t, loc[:, 0], loc[:, 1]), bins=(t_bins,) + loc_bins)
        raw = fac * raw / (nrns_per_bin + 1E-6)
        return raw
    return spiking3dhistogram


# noinspection PyStatementEffect
def save(Hs, t_bins, loc_bins, out_root):
    if not os.path.isdir(out_root):
        _ = os.makedirs(out_root)
    import h5py
    h5 = h5py.File(os.path.join(out_root, "spiking_activity_3d.h5"), "w")
    grp_bins = h5.create_group("bins")
    grp_bins.create_dataset("t", data=t_bins)
    grp_bins.create_dataset("x", data=loc_bins[0])
    grp_bins.create_dataset("y", data=loc_bins[1])

    grp_data = h5.create_group("histograms")
    for i, val in enumerate(Hs):
        grp_data.create_dataset("instance{0}".format(i), data=val)
    mn_data = numpy.mean(numpy.stack(Hs, -1), axis=-1)
    grp_data.create_dataset("mean", data=mn_data)
    return mn_data


# noinspection PyStatementEffect
def plot(mn_hist, t_bins, loc_bins, out_root):
    from matplotlib import pyplot as plt
    mx_clim = numpy.percentile(mn_hist[mn_hist > 0], 99)
    i = 0
    for t_start, t_end, m in tqdm.tqdm(zip(t_bins[:-1], t_bins[1:], mn_hist)):
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        img = ax.imshow(m)
        img.set_clim([0, mx_clim])
        ax.set_title("{0} - {1} ms".format(t_start, t_end))
        plt.colorbar(img)
        plt.box(False)
        fn = "frame{:04d}.png".format(i)
        i += 1
        fig.savefig(os.path.join(out_root, fn))
        plt.close(fig)


def main():
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logformat)
    np.random.seed(0)
    etl.register_accessors()
    sims, cfg = parse_arguments()
    out_fn = sys.argv[3]
    data, circ_dict = initial_setup(sims)

    assert len(circ_dict) == 1, "Simulations must use the same circuit"
    circ = list(circ_dict.values())[0]

    # Specify in config to apply the analysis only to a subset of simulation conditions
    data = data.etl.filter(**cfg.get("condition_filter", {}), drop_level=False)
    spikes = data.map(read_spikes)

    # time bins
    t_bins = make_t_bins(sims, cfg.get("t_start", None), cfg.get("t_end", None), cfg.get("t_step", 20.0))

    # Get neuron locations and bins
    gids = numpy.unique(numpy.hstack(data.map(get_sim_gids)))
    locations = circ.cells.get(gids, ["x", "y", "z"])
    flat_locations = flatten_locations(locations, cfg["flatmap"])
    loc_bins = make_bins(flat_locations, cfg.get("nbins", 1000))

    # Build histogramming function and apply
    func = make_histogram_function(t_bins, loc_bins, flat_locations)
    Hs = spikes.map(func)
    mn_data = save(Hs, t_bins, loc_bins, out_fn)
    plot(mn_data, t_bins, loc_bins, out_fn)


if __name__ == "__main__":
    main()
