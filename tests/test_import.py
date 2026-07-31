import motorch


def test_motorch_import_and_version() -> None:
    assert hasattr(motorch, "__version__")
    assert motorch.__version__ == "0.0.1"
