# Basic test case to test functioning of module's top-level

try:
    import blueetl  # noqa

    _top_import_error = None
except Exception as ex:  # pragma: no cover
    _top_import_error = ex


def test_import():
    assert _top_import_error is None
