import pytest
from plotsense.exceptions import PlotSenseError, PlotSenseAPIError, PlotSenseDataError, PlotSenseConfigError


# Testing inheritance
def test_exception_inheritance():
    assert isinstance(PlotSenseAPIError("API failed"), ConnectionError)
    assert isinstance(PlotSenseDataError("Invalid data"), ValueError)
    assert isinstance(PlotSenseConfigError("Missing config"), KeyError)
    assert isinstance(PlotSenseError("General error"), Exception)

# Raising and catching each exception


def test_exception_raises():
    with pytest.raises(PlotSenseAPIError):
        raise PlotSenseAPIError("API error occurred")
    with pytest.raises(PlotSenseDataError):
        raise PlotSenseDataError("Data error occurred")
    with pytest.raises(PlotSenseConfigError):
        raise PlotSenseConfigError("Config error occurred")
    with pytest.raises(PlotSenseError):
        raise PlotSenseError("error occurred")

# Testing exception messages


def test_exception_messages():
    api_error = PlotSenseAPIError("API error occurred")
    data_error = PlotSenseDataError("Data error occurred")
    config_error = PlotSenseConfigError("Config error occurred")
    general_error = PlotSenseError("General error occurred")

    assert str(api_error) == "API error occurred"
    assert str(data_error) == "Data error occurred"
    assert str(config_error) == "Config error occurred"
    assert str(general_error) == "General error occurred"
