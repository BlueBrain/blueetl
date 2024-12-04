import a01.run as test_module
from common.utils import load_json

SIMULATION_CAMPAIGN = "/gpfs/bbp.cscs.ch/data/scratch/proj30/home/ficarell/SBO/sims/249e64ca-58fc-4d11-9c2d-b5bdbcc1dc9d_2/config.json"


def test_main(tmp_path):
    analysis_output = tmp_path / "analysis_output.json"
    result = test_module.main(
        analysis_config={
            "simulation_campaign": SIMULATION_CAMPAIGN,
            "output": tmp_path,
            "report_type": "spikes",
            "report_name": "raster",
            "node_sets": ["Inhibitory", "Excitatory"],
            "cell_step": 1,
        },
        analysis_output=analysis_output,
    )
    expected = {
        "outputs": [
            {
                "cell_step": 1,
                "node_sets": ["Inhibitory", "Excitatory"],
                "path": str(tmp_path / "plot_spikes_raster_0.png"),
                "report_name": "raster",
                "report_type": "spikes",
                "simulation_ids": [0],
            },
            {
                "cell_step": 1,
                "node_sets": ["Inhibitory", "Excitatory"],
                "path": str(tmp_path / "plot_spikes_raster_1.png"),
                "report_name": "raster",
                "report_type": "spikes",
                "simulation_ids": [1],
            },
        ]
    }
    assert result == expected
    assert analysis_output.exists()
    assert load_json(analysis_output) == expected
