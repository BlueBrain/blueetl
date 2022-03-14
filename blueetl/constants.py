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
TRIAL = "trial"
T_START = "t_start"
T_STOP = "t_stop"
DURATION = "duration"
COUNT = "count"

DTYPES = {
    GID: np.int64,
    TIME: np.float64,
    SIMULATION_ID: np.int16,
    CIRCUIT_ID: np.int16,
    NEURON_CLASS: "category",
    WINDOW: "category",
    TRIAL: np.int16,
    T_START: np.float64,
    T_STOP: np.float64,
    DURATION: np.float64,
}
# TODO: category or object?
