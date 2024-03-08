from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.emodel_pipeline import plotting


def get_local_access_point(metadata):
    return LocalAccessPoint(
        emodel=metadata["emodel"],
        emodel_dir=metadata["path"],
        etype=metadata["etype"],
        iteration_tag=metadata["iteration"],
        recipes_path="recipes.json",
    )


def get_recording_names(access_point, cell_evaluator):
    return plotting.get_recording_names(
        protocol_config=access_point.get_fitness_calculator_configuration().protocols,
        stimuli=cell_evaluator.fitness_protocols["main_protocol"].protocols,
    )
