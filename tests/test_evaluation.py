from latedeliveries.evaluation import order_cost


def test_example_from_assignment():
    y_true = [1, 0, 0, 1, 0]

    flagged = [
        True,
        True,
        False,
        False,
        False,
    ]

    assert order_cost(y_true, flagged) == 23


def test_flag_everybody():
    y_true = [1, 1, 0, 0]

    flagged = [True, True, True, True]

    assert order_cost(y_true, flagged) == 16


def test_flag_nobody():
    y_true = [1, 1, 0, 0]

    flagged = [False, False, False, False]

    assert order_cost(y_true, flagged) == 30