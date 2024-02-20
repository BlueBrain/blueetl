from pathlib import Path

import neuron
from bluecellulab import Cell, Simulation
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.stimuli import ShotNoise
from common.utils import L, load_json, run_analysis
from matplotlib import pyplot as plt


def _savefig(path, title):
    """Save a matplotlib figure."""
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close("all")
    plt.clf()


def _get_one(path, patterns, raise_if_multiple=True):
    found = []
    for pattern in patterns:
        found.extend(Path(path).glob(pattern))
    if len(found) == 0 and raise_if_multiple:
        raise FileNotFoundError(f"No matching files found in {path}")
    if len(found) > 1:
        raise Exception(f"Multiple matching files found in {path}")
    return found[0]


def _get_hoc(path):
    return Path(path) / "model.hoc"


def _get_morphology(path, file_extensions=(".asc", ".swc")):
    return _get_one(path, patterns=["*" + ext for ext in file_extensions])


def _get_emodel(path):
    return _get_one(path, patterns=["EM_*.json"])


def _get_emodel_properties(emodel_data):
    holding_current = None
    threshold_current = None
    for feature in emodel_data["features"]:
        if "bpo_holding_current" in feature["name"]:
            holding_current = feature["value"]
            L.info(feature)
        elif "bpo_threshold_current" in feature["name"]:
            threshold_current = feature["value"]
            L.info(feature)
    return EmodelProperties(
        threshold_current=threshold_current,
        holding_current=holding_current,
    )


def _run_simulation(cell, max_time):
    """Run the simulation."""
    sim = Simulation()
    sim.add_cell(cell)
    sim.run(max_time, cvode=False, v_init=-75)
    return cell.get_time(), cell.get_soma_voltage()


def _step_stimulus(hoc_file, morph_file, emodel_properties, step_parameters):
    """Define stimulus parameters."""
    cell = Cell(hoc_file, morph_file, template_format="v6", emodel_properties=emodel_properties)
    icneurodamusobj = cell.add_step(
        start_time=step_parameters["start_time"],
        stop_time=step_parameters["stop_time"],
        level=step_parameters["level"],
    )
    iclamp_current = neuron.h.Vector()
    iclamp_current.record(icneurodamusobj.ic._ref_i)
    return cell, iclamp_current


def _step_plot(time, voltage, current):
    """Plot the simulation."""
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 10))

    ax1.plot(time, current, drawstyle="steps-post")
    # ax1.plot(time_vec, stim_vec, drawstyle='steps-post')
    ax1.set_title("Stimulus")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Current (nA)")
    # ax1.fill_between(time_vec, 0, stim_vec, step="post", color='gray', alpha=0.3)
    ax1.fill_between(time, 0, current, step="post", color="gray", alpha=0.3)
    ax1.grid(True)

    ax2.plot(time, voltage)
    ax2.set_title("Response")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Voltage (mV)")
    ax2.grid(True)  # If you want grid on the second plot as well


def _ramp_stimulus(hoc_file, morph_file, emodel_properties, ramp_parameters):
    """Define ramp stimulus parameters."""
    cell = Cell(hoc_file, morph_file, template_format="v6", emodel_properties=emodel_properties)
    ramp_obj = cell.add_ramp(
        start_time=ramp_parameters["start_time"],
        stop_time=ramp_parameters["stop_time"],
        start_level=ramp_parameters["start_level"],
        stop_level=ramp_parameters["stop_level"],
    )

    ramp_current = neuron.h.Vector()
    ramp_current.record(ramp_obj.ic._ref_i)

    # To add the holding current
    # from bluecellulab.cell.injector import Hyperpolarizing
    # hyperpolarizing = Hyperpolarizing("single-cell", delay=0, duration=params['tstop'])
    # cell.add_replay_hypamp(hyperpolarizing)

    return cell, ramp_current


def _ramp_plot(time, voltage, current, ramp_parameters):
    """Plot the simulation."""
    # time_vec = [0, start_time, stop_time, max_time]
    # stim_vec = [0, start_level, stop_level, stop_level]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 10))

    ax1.plot(time, current)
    # ax1.plot(time_vec, stim_vec, '-o')
    ax1.set_title("Ramp Stimulus")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Current (nA)")
    ax1.fill_between(
        [ramp_parameters["start_time"], ramp_parameters["stop_time"]],
        ramp_parameters["start_level"],
        ramp_parameters["stop_level"],
        color="gray",
        alpha=0.3,
    )
    ax1.grid(True)

    ax2.plot(time, voltage)
    ax2.set_title("Response")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Voltage (mV)")
    ax2.grid(True)


def _shotnoise_stimulus(hoc_file, morph_file, emodel_properties, shotnoise_parameters):
    """Define stimulus parameters."""
    shotnoise_stimulus = ShotNoise(target="single-cell", **shotnoise_parameters)
    cell = Cell(hoc_file, morph_file, template_format="v6", emodel_properties=emodel_properties)
    time_vec, stim_vec = cell.add_replay_shotnoise(
        cell.soma, 0.5, shotnoise_stimulus, shotnoise_stim_count=3
    )
    time_vec = time_vec.to_python()
    stim_vec = stim_vec.to_python()
    return cell, time_vec, stim_vec


def _shotnoise_plot(time, voltage, time_vec, stim_vec, max_time):
    """Plot the simulation."""
    new_stim_vec = [0] + stim_vec + [0]
    new_time_vec = [0] + time_vec + [max_time]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

    ax1.plot(new_time_vec, new_stim_vec, "-o")
    # ax1.plot(time_vec, stim_vec, '-o')
    ax1.set_title("Shot noise Stimulus")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Current (nA)")
    ax1.grid(True)

    ax2.plot(time, voltage)
    ax2.set_title("Response")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Voltage (mV)")
    ax2.grid(True)


def _step_analysis(hoc_file, morph_file, emodel_properties, output_file):
    max_time = 600
    step_parameters = {
        "start_time": 50.0,  # Start time of the stimulus
        "stop_time": 500.0,  # Stop time of the stimulus
        "level": 0.6,  # Current level of the stimulus
    }
    cell, current = _step_stimulus(hoc_file, morph_file, emodel_properties, step_parameters)
    time, voltage = _run_simulation(cell, max_time=max_time)
    _step_plot(time, voltage, current)
    _savefig(output_file, title="Step")


def _ramp_analysis(hoc_file, morph_file, emodel_properties, output_file):
    max_time = 200
    ramp_parameters = {
        "start_time": 50.0,  # Start time of the ramp
        "stop_time": 125.0,  # Stop time of the ramp
        "start_level": 0.0,  # Start level of the ramp
        "stop_level": 2.0,  # Stop level of the ramp
    }
    cell, current = _ramp_stimulus(hoc_file, morph_file, emodel_properties, ramp_parameters)
    time, voltage = _run_simulation(cell, max_time=max_time)
    _ramp_plot(time, voltage, current, ramp_parameters)
    _savefig(output_file, title="Ramp")


def _shotnoise_analysis(hoc_file, morph_file, emodel_properties, output_file):
    max_time = 60
    shotnoise_parameters = {
        "delay": 25,
        "duration": 20,
        "rise_time": 0.4,
        "decay_time": 4,
        "rate": 2e3,
        "amp_mean": 40e-3,
        "amp_var": 16e-4,
        "seed": 3899663,
    }
    cell, time_vec, stim_vec = _shotnoise_stimulus(
        hoc_file, morph_file, emodel_properties, shotnoise_parameters
    )
    time, voltage = _run_simulation(cell, max_time=max_time)
    _shotnoise_plot(time, voltage, time_vec, stim_vec, max_time=max_time)
    _savefig(output_file, title="Shot Noise")


@run_analysis
def main(analysis_config: dict) -> dict:
    L.info("analysis_config:\n%s", analysis_config)
    path = Path(analysis_config["emodel"]["path"])
    path = path / "nexus_temp/emodel=cSTUT__iteration=2f92aa0"  # FIXME
    output_dir = Path(analysis_config["output"])
    outputs = []

    # Analysis based on obp_emodel_localrun.zip found at
    # https://bbpteam.epfl.ch/project/issues/browse/BBPP134-1367#comment-234746
    hoc_file = _get_hoc(path)
    morph_file = _get_morphology(path)
    emodel_file = _get_emodel(path)
    emodel_data = load_json(emodel_file)
    emodel_properties = _get_emodel_properties(emodel_data)

    output_file = output_dir / "step.pdf"
    _step_analysis(hoc_file, morph_file, emodel_properties, output_file=output_file)
    outputs.append(
        {
            "path": str(output_file),
            "report_type": "step",
            "report_name": "Step Stimulus Analysis",
        }
    )

    output_file = output_dir / "ramp.pdf"
    _ramp_analysis(hoc_file, morph_file, emodel_properties, output_file=output_file)
    outputs.append(
        {
            "path": str(output_file),
            "report_type": "ramp",
            "report_name": "Ramp Stimulus Analysis",
        }
    )

    output_file = output_dir / "shotnoise.pdf"
    _shotnoise_analysis(hoc_file, morph_file, emodel_properties, output_file=output_file)
    outputs.append(
        {
            "path": str(output_file),
            "report_type": "shotnoise",
            "report_name": "Shot Noise Stimulus Analysis",
        }
    )

    return {"outputs": outputs}
