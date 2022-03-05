import numpy as np

GID = "gid"
TIME = "time"
SIMULATION_PATH = "simulation_path"
SIMULATION_ID = "simulation_id"
SIMULATION = "simulation"
CIRCUIT_ID = "circuit_id"
CIRCUIT = "circuit"
NEURON_CLASS = "neuron_class"
WINDOW = "window"
COUNT = "count"

DTYPES = {
    GID: np.int64,
    TIME: np.float64,
    SIMULATION_ID: np.int8,
    CIRCUIT_ID: np.int8,
    NEURON_CLASS: "category",
    WINDOW: "category",
}
# TODO: category or object?
