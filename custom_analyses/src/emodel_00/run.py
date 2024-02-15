import os
from pathlib import Path

from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.evaluation import evaluation
from bluepyemodelnexus.nexus import NexusAccessPoint
from common.utils import L, run_analysis

NEXUS_FORGE_PATH = "https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml"


def _get_nexus_access_point(metadata):
    return NexusAccessPoint(
        emodel=metadata["emodel"],
        etype=metadata["etype"],
        iteration_tag=metadata["iteration"],
        project=os.getenv("NEXUS_PROJ"),
        organisation=os.getenv("NEXUS_ORG"),
        endpoint=os.getenv("NEXUS_BASE"),
        access_token=os.getenv("NEXUS_TOKEN"),
        forge_path=NEXUS_FORGE_PATH,
    )


def _get_recording_names(access_point, cell_evaluator):
    return plotting.get_recording_names(
        protocol_config=access_point.get_fitness_calculator_configuration().protocols,
        stimuli=cell_evaluator.fitness_protocols["main_protocol"].protocols,
    )


@run_analysis
def main(analysis_config: dict) -> dict:
    L.info("analysis_config:\n%s", analysis_config)
    L.info("NEXUS_BASE=%s", os.getenv("NEXUS_BASE"))
    L.info("NEXUS_ORG=%s", os.getenv("NEXUS_ORG"))
    L.info("NEXUS_PROJ=%s", os.getenv("NEXUS_PROJ"))
    access_point = _get_nexus_access_point(analysis_config["emodel"])
    emodel = access_point.get_emodel()
    cell_evaluator = evaluation.get_evaluator_from_access_point(
        access_point,
        stochasticity=False,
        include_validation_protocols=False,
    )
    L.info("cell_evaluator:\n%s", cell_evaluator)
    recording_names = _get_recording_names(access_point, cell_evaluator)
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
