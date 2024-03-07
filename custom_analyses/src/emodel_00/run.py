import os
from pathlib import Path

from common.utils import L, run_analysis
from emodel_00.lib import EModelAccessor, RampAnalysis, ShotNoiseAnalysis, StepAnalysis, run_all


@run_analysis
def main(analysis_config: dict) -> dict:
    """Simple analysis example."""
    L.info("analysis_config:\n%s", analysis_config)
    L.info("Working directory: %s", os.getcwd())
    L.info("BLUECELLULAB_MOD_LIBRARY_PATH=%s", os.getenv("BLUECELLULAB_MOD_LIBRARY_PATH"))
    accessor = EModelAccessor.from_metadata(analysis_config["emodel"])
    output_dir = Path(analysis_config["output"])

    analyses = [
        StepAnalysis(
            accessor=accessor,
            output_file=output_dir / "step.pdf",
            stimulus_parameters={
                "start_time": 50.0,  # Start time of the stimulus
                "stop_time": 500.0,  # Stop time of the stimulus
                "level": 0.6,  # Current level of the stimulus
            },
            simulation_parameters={
                "maxtime": 600,
                "cvode": False,
                "v_init": -75,
            },
        ),
        RampAnalysis(
            accessor=accessor,
            output_file=output_dir / "ramp.pdf",
            stimulus_parameters={
                "start_time": 50.0,  # Start time of the ramp
                "stop_time": 125.0,  # Stop time of the ramp
                "start_level": 0.0,  # Start level of the ramp
                "stop_level": 2.0,  # Stop level of the ramp
            },
            simulation_parameters={
                "maxtime": 200,
                "cvode": False,
                "v_init": -75,
            },
        ),
        ShotNoiseAnalysis(
            accessor,
            output_file=output_dir / "shotnoise.pdf",
            stimulus_parameters={
                "delay": 25,
                "duration": 20,
                "rise_time": 0.4,
                "decay_time": 4,
                "rate": 2e3,
                "amp_mean": 40e-3,
                "amp_var": 16e-4,
                "seed": 3899663,
            },
            simulation_parameters={
                "maxtime": 60,
                "cvode": False,
                "v_init": -75,
            },
        ),
    ]

    run_all(analyses)
    return {"outputs": [a.metadata() for a in analyses]}
