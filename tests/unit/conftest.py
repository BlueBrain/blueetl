import pytest


@pytest.fixture
def lazy_fixture(request):
    """Return a function that returns the actual fixture from its name."""
    return lambda name: request.getfixturevalue(name)
