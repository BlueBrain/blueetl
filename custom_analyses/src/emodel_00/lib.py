import abc
import dataclasses
from functools import cached_property
from pathlib import Path
from typing import Any

from common.utils import L, isolated, load_json
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

DEFAULT_CELL_TEMPLATE = "v6"
DEFAULT_SIMULATION_TIMEOUT = 3600


@dataclasses.dataclass(kw_only=True)
class EModelAccessor:
    """EModelAccessor."""

    emodel: str
    etype: str
    iteration: str
    seed: int
    recipes_file: Path
    hoc_file: Path
    morph_file: Path
    emodel_properties: dict[str, Any]

    @staticmethod
    def _get_emodel_properties(emodel_data):
        """Load and return the EmodelProperties."""
        holding_current = 0.0
        threshold_current = 0.0
        for feature in emodel_data["features"]:
            if "bpo_holding_current" in feature["name"]:
                holding_current = feature["value"]
                L.info(feature)
            elif "bpo_threshold_current" in feature["name"]:
                threshold_current = feature["value"]
                L.info(feature)
        return {
            "threshold_current": threshold_current,
            "holding_current": holding_current,
        }

    @classmethod
    def from_metadata(cls, metadata):
        emodel_dir = Path(metadata["path"])
        hoc_file = emodel_dir / "model.hoc"
        recipes_file = emodel_dir / "recipes.json"
        recipes = load_json(recipes_file)[metadata["emodel"]]
        morphology_file = emodel_dir / recipes["morph_path"] / recipes["morphology"]
        emodel_properties = cls._get_emodel_properties(load_json(emodel_dir / recipes["final"]))
        return cls(
            emodel=metadata["emodel"],
            etype=metadata["etype"],
            iteration=metadata["iteration"],
            seed=metadata["seed"],
            recipes_file=recipes_file,
            hoc_file=hoc_file,
            morph_file=morphology_file,
            emodel_properties=emodel_properties,
        )


class BaseAnalysis(abc.ABC):
    """BaseAnalysis class."""

    report_type = None
    report_name = None

    def __init__(
        self,
        accessor,
        output_file,
        stimulus_parameters,
        simulation_parameters,
        timeout=DEFAULT_SIMULATION_TIMEOUT,
    ):
        """Init the Analysis."""
        assert "maxtime" in simulation_parameters, "maxtime is a required property"
        self.accessor = accessor
        self.output_file = output_file
        self.stimulus_parameters = stimulus_parameters
        self.simulation_parameters = simulation_parameters
        self.timeout = timeout

    def metadata(self):
        """Return the analysis metadata."""
        return {
            "path": str(self.output_file),
            "report_type": self.report_type,
            "report_name": self.report_name,
        }

    @abc.abstractmethod
    def run(self):
        """Run the analysis, to be implemented in the subclass."""

    @cached_property
    def _bluecellulab(self):
        # pylint: disable=import-outside-toplevel
        import bluecellulab

        return bluecellulab

    @cached_property
    def _neuron(self):
        # pylint: disable=import-outside-toplevel
        import neuron

        return neuron

    def _get_cell(self, *, template_format=DEFAULT_CELL_TEMPLATE, **kwargs):
        """Instantiate and return a new Cell."""
        emodel_properties = self._bluecellulab.EmodelProperties(**self.accessor.emodel_properties)
        return self._bluecellulab.Cell(
            self.accessor.hoc_file,
            self.accessor.morph_file,
            template_format=template_format,
            emodel_properties=emodel_properties,
            **kwargs,
        )

    def _run_simulation(self, cell):
        """Instantiate, run, and return a new Simulation."""
        sim = self._bluecellulab.Simulation()
        sim.add_cell(cell)
        sim.run(**self.simulation_parameters)
        return sim


class StepAnalysis(BaseAnalysis):

    report_type = "step"
    report_name = "Step Stimulus Analysis"

    def run(self):
        cell = self._get_cell()
        tstim = cell.add_step(
            start_time=self.stimulus_parameters["start_time"],
            stop_time=self.stimulus_parameters["stop_time"],
            level=self.stimulus_parameters["level"],
        )
        stim_current = self._neuron.h.Vector()
        stim_current.record(tstim.ic._ref_i)
        self._run_simulation(cell)
        time, voltage = cell.get_time(), cell.get_soma_voltage()
        self._plot(time, voltage, stim_current.to_python())
        savefig(self.output_file, title=self.report_name)

    def _plot(self, time, voltage, current):
        """Plot the simulation."""
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


class RampAnalysis(BaseAnalysis):

    report_type = "ramp"
    report_name = "Ramp Stimulus Analysis"

    def run(self):
        cell = self._get_cell()
        tstim = cell.add_ramp(
            start_time=self.stimulus_parameters["start_time"],
            stop_time=self.stimulus_parameters["stop_time"],
            start_level=self.stimulus_parameters["start_level"],
            stop_level=self.stimulus_parameters["stop_level"],
        )
        stim_current = self._neuron.h.Vector()
        stim_current.record(tstim.ic._ref_i)
        self._run_simulation(cell)
        time, voltage = cell.get_time(), cell.get_soma_voltage()
        self._plot(time, voltage, stim_current.to_python())
        savefig(self.output_file, title=self.report_name)

    def _plot(self, time, voltage, current):
        """Plot the simulation."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 10))

        ax1.plot(time, current)
        ax1.set_title("Ramp Stimulus")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Current (nA)")
        ax1.fill_between(
            [self.stimulus_parameters["start_time"], self.stimulus_parameters["stop_time"]],
            self.stimulus_parameters["start_level"],
            self.stimulus_parameters["stop_level"],
            color="gray",
            alpha=0.3,
        )
        ax1.grid(True)

        ax2.plot(time, voltage)
        ax2.set_title("Response")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Voltage (mV)")
        ax2.grid(True)


class ShotNoiseAnalysis(BaseAnalysis):

    report_type = "shotnoise"
    report_name = "Shot Noise Stimulus Analysis"

    def run(self):
        cell = self._get_cell(rng_settings=self._bluecellulab.RNGSettings(base_seed=0))
        shotnoise_stimulus = self._bluecellulab.stimuli.ShotNoise(
            target="single-cell",
            **self.stimulus_parameters,
        )
        time_vec, stim_vec = cell.add_replay_shotnoise(
            section=cell.soma,
            segx=0.5,
            stimulus=shotnoise_stimulus,
            shotnoise_stim_count=3,
        )
        self._run_simulation(cell)
        time, voltage = cell.get_time(), cell.get_soma_voltage()
        self._plot(time, voltage, time_vec.to_python(), stim_vec.to_python())
        savefig(self.output_file, title=self.report_name)

    def _plot(self, time, voltage, time_vec, stim_vec):
        """Plot the simulation."""
        new_stim_vec = [0] + stim_vec + [0]
        new_time_vec = [0] + time_vec + [self.simulation_parameters["maxtime"]]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

        ax1.plot(new_time_vec, new_stim_vec, "-o")
        ax1.set_title("Shot noise Stimulus")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Current (nA)")
        ax1.grid(True)

        ax2.plot(time, voltage)
        ax2.set_title("Response")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Voltage (mV)")
        ax2.grid(True)


def savefig(path, title):
    """Save a matplotlib figure."""
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close("all")
    plt.clf()


def run_all(analyses, n_jobs=4, verbose=10):
    """Run all the analyses in separate processes to ensure that neuron is isolated."""
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    return parallel(delayed(isolated(a.run))() for a in analyses)
