from latedeliveries.pipeline import (
    CATEGORICAL,
    NUMERIC,
    build_preprocessor,
)


def test_numeric_columns():
    assert len(NUMERIC) == 4


def test_categorical_columns():
    assert len(CATEGORICAL) == 3


def test_build_preprocessor():
    preprocessor = build_preprocessor()

    assert preprocessor is not None