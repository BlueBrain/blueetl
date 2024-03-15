from click.testing import CliRunner

from blueetl.apps import migrate as test_module
from blueetl.constants import CONFIG_VERSION
from blueetl.utils import dump_yaml, load_yaml


def test_migrate_config(tmp_path):
    input_config_file = "input_config.yaml"
    output_config_file = "output_config.yaml"
    input_data = {
        "simulation_campaign": "/path/to/config.json",
        "simulations_filter_in_memory": {"simulation_id": 2},
        "output": "output_dir",
        "extraction": {
            "neuron_classes": {
                "L1_EXC": {"layer": ["1"], "synapse_class": ["EXC"]},
                "L1_EXC_gids": {"layer": ["1"], "synapse_class": ["EXC"], "gid": [1, 2]},
            },
            "limit": None,
            "target": None,
            "windows": {"w1": {"bounds": [20, 90], "window_type": "spontaneous"}},
        },
        "analysis": {
            "features": [
                {
                    "type": "multi",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "function": "module.user.function",
                    "params": {"export_all_neurons": True},
                }
            ]
        },
    }
    expected_data = {
        "version": CONFIG_VERSION,
        "simulation_campaign": "/path/to/config.json",
        "simulations_filter_in_memory": {"simulation_id": 2},
        "output": "output_dir",
        "analysis": {
            "spikes": {
                "extraction": {
                    "report": {"type": "spikes"},
                    "neuron_classes": {
                        "L1_EXC": {"query": {"layer": ["1"], "synapse_class": ["EXC"]}},
                        "L1_EXC_gids": {
                            "query": {"layer": ["1"], "synapse_class": ["EXC"]},
                            "node_id": [1, 2],
                        },
                    },
                    "limit": None,
                    "node_set": None,
                    "windows": {"w1": {"bounds": [20, 90], "window_type": "spontaneous"}},
                },
                "features": [
                    {
                        "type": "multi",
                        "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                        "function": "module.user.function",
                        "params": {"export_all_neurons": True},
                    }
                ],
            }
        },
    }
    expected_message = "The converted configuration has been saved to output_config.yaml."
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        dump_yaml(input_config_file, input_data)
        result = runner.invoke(test_module.migrate_config, [input_config_file, output_config_file])
        assert result.exit_code == 0
        assert result.output.strip() == expected_message
        output_data = load_yaml(output_config_file)

    assert output_data == expected_data
