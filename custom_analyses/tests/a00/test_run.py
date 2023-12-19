import a00.run as test_module


def test_main(tmp_path):
    result = test_module.main(
        analysis_config={
            "simulation_campaign": "/path/to/config.json",
            "output": tmp_path,
            "report_type": "spikes",
            "report_name": "raster",
            "node_sets": ["Inhibitory", "Excitatory"],
            "cell_step": 1,
        },
    )
    assert result == {"outputs": []}
