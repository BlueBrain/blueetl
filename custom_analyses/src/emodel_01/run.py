import os
from pathlib import Path

from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.evaluation import evaluation
from common.utils import L, cwd, run_analysis
from emodel_01.lib import get_local_access_point, get_recording_names

# Example of analysis_config:
# {
#   "emodel": {
#     "id": "<Nexus id>",
#     "url": "<Nexus url>",
#     "path": "/path/to/emodel/dir",
#     "emodel": "cSTUT",
#     "etype": "cSTUT",
#     "iteration": "2f92aa0",
#     "seed": "11"
#   },
#   "output": "/path/to/scratch/dir"
# }


@run_analysis
def main(analysis_config: dict) -> dict:
    """Simple analysis example using bluepyemodel."""
    L.info("analysis config:\n%s", analysis_config)
    # to work with BluePyEModel we need to change the working directory
    with cwd(analysis_config["emodel"]["path"]):
        L.info("working directory: %s", os.getcwd())
        access_point = get_local_access_point(analysis_config["emodel"])
        emodel = access_point.get_emodel()
        cell_evaluator = evaluation.get_evaluator_from_access_point(
            access_point,
            stochasticity=False,
            include_validation_protocols=False,
        )
        L.info("cell_evaluator:\n%s", cell_evaluator)
        recording_names = get_recording_names(access_point, cell_evaluator)
        responses = cell_evaluator.run_protocols(
            protocols=cell_evaluator.fitness_protocols.values(),
            param_values=emodel.parameters,
        )
        figures_dir = Path(analysis_config["output"]) / "figures"
        plotting.traces(
            model=emodel,
            responses=responses,
            recording_names=recording_names,
            figures_dir=figures_dir,
        )
        return {"outputs": [{"path": str(f)} for f in figures_dir.iterdir() if f.is_file()]}
