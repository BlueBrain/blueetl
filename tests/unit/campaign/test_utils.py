import pytest

from blueetl.campaign import utils as test_module


@pytest.mark.parametrize(
    "include_empty, expected",
    [
        (
            True,
            [
                ({"ca": 1.0, "depolarization": 3.0}, None),
                ({"ca": 1.0, "depolarization": 4.0}, "/tmp/simple/uuid/1/simulation_config.json"),
                ({"ca": 2.0, "depolarization": 3.0}, "/tmp/simple/uuid/2/simulation_config.json"),
                ({"ca": 2.0, "depolarization": 4.0}, "/tmp/simple/uuid/3/simulation_config.json"),
            ],
        ),
        (
            False,
            [
                ({"ca": 1.0, "depolarization": 4.0}, "/tmp/simple/uuid/1/simulation_config.json"),
                ({"ca": 2.0, "depolarization": 3.0}, "/tmp/simple/uuid/2/simulation_config.json"),
                ({"ca": 2.0, "depolarization": 4.0}, "/tmp/simple/uuid/3/simulation_config.json"),
            ],
        ),
    ],
)
def test_campaign_sims(xarray_config_dict_simple, include_empty, expected):
    result = list(test_module.campaign_sims(xarray_config_dict_simple, include_empty=include_empty))
    assert result == expected


@pytest.mark.parametrize(
    "include_empty, expected",
    [
        (
            True,
            [
                ({"ca": 1.0, "depolarization": 3.0}, None),
                ({"ca": 2.0, "depolarization": 4.0}, "/tmp/coupled/uuid/1/simulation_config.json"),
            ],
        ),
        (
            False,
            [
                ({"ca": 2.0, "depolarization": 4.0}, "/tmp/coupled/uuid/1/simulation_config.json"),
            ],
        ),
    ],
)
def test_campaign_sims_coupled(xarray_config_dict_coupled, include_empty, expected):
    result = list(
        test_module.campaign_sims(xarray_config_dict_coupled, include_empty=include_empty)
    )
    assert result == expected


def test_campaign_sims_with_indices(xarray_config_dict_simple):
    result = list(test_module._campaign_sims_with_indices(xarray_config_dict_simple))
    expected = [
        (
            {"ca": 1.0, "depolarization": 3.0},
            {
                "ca_idx": 0,
                "ca_total": 2,
                "depolarization_idx": 0,
                "depolarization_total": 2,
                "idx": 0,
            },
            "",
        ),
        (
            {"ca": 1.0, "depolarization": 4.0},
            {
                "ca_idx": 0,
                "ca_total": 2,
                "depolarization_idx": 1,
                "depolarization_total": 2,
                "idx": 1,
            },
            "/tmp/simple/uuid/1/simulation_config.json",
        ),
        (
            {"ca": 2.0, "depolarization": 3.0},
            {
                "ca_idx": 1,
                "ca_total": 2,
                "depolarization_idx": 0,
                "depolarization_total": 2,
                "idx": 2,
            },
            "/tmp/simple/uuid/2/simulation_config.json",
        ),
        (
            {"ca": 2.0, "depolarization": 4.0},
            {
                "ca_idx": 1,
                "ca_total": 2,
                "depolarization_idx": 1,
                "depolarization_total": 2,
                "idx": 3,
            },
            "/tmp/simple/uuid/3/simulation_config.json",
        ),
    ]
    assert result == expected


def test_campaign_sims_with_indices_coupled(xarray_config_dict_coupled):
    with pytest.raises(ValueError, match="Please provide non-coupled coords sim campaign"):
        list(test_module._campaign_sims_with_indices(xarray_config_dict_coupled))


def test_campaign_sim_indices(xarray_config_dict_simple):
    result = test_module._campaign_sim_indices(xarray_config_dict_simple)
    assert result == [1, 2, 3]


def test_campaign_sim_indices_coupled(xarray_config_dict_coupled):
    result = test_module._campaign_sim_indices(xarray_config_dict_coupled)
    assert result == [1]


def test_campaign_sim_index_to_coords(xarray_config_dict_simple):
    result = test_module._campaign_sim_index_to_coords(xarray_config_dict_simple)
    assert result == {
        1: {"ca": 1.0, "depolarization": 4.0},
        2: {"ca": 2.0, "depolarization": 3.0},
        3: {"ca": 2.0, "depolarization": 4.0},
    }


def test_campaign_sim_index_to_coords_coupled(xarray_config_dict_coupled):
    result = test_module._campaign_sim_index_to_coords(xarray_config_dict_coupled)
    assert result == {
        1: {"ca": 2.0, "depolarization": 4.0},
    }
