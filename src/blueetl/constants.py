"""Common constants."""

import numpy as np

# columns names
GID = "gid"
TIME = "time"
SIMULATION_PATH = "simulation_path"
SIMULATION_ID = "simulation_id"
SIMULATION = "simulation"
CIRCUIT_ID = "circuit_id"
CIRCUIT = "circuit"
NEURON_CLASS = "neuron_class"
NEURON_CLASS_INDEX = "neuron_class_index"  # incremental gid index for each neuron class
WINDOW = "window"
TRIAL = "trial"
OFFSET = "offset"
T_START = "t_start"
T_STOP = "t_stop"
T_STEP = "t_step"
DURATION = "duration"
WINDOW_TYPE = "window_type"
COUNT = "count"
TIMES = "times"
BIN = "bin"
VALUE = "value"
SECTION = "section"
LIMIT = "limit"
POPULATION = "population"
NODE_SET = "node_set"
GIDS = "gids"
QUERY = "query"

DTYPES = {
    GID: np.int64,
    TIME: np.float64,
    SIMULATION_ID: np.int16,
    CIRCUIT_ID: np.int16,
    NEURON_CLASS: "category",
    WINDOW: "category",
    TRIAL: np.int16,
    OFFSET: np.float64,
    T_START: np.float64,
    T_STOP: np.float64,
    T_STEP: np.float64,
    DURATION: np.float64,
}
CHECKSUM_SEP = "#"
LEVEL_SEP = "."
CONFIG_VERSION = 3
