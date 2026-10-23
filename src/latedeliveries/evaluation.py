def order_cost(y_true, flagged):
    """
    Company cost rule:

    Flagged order:
        $4 voucher cost

    Late order not flagged:
        $15 support cost
    """

    total_cost = 0

    for actual, flag in zip(y_true, flagged):
        if flag:
            total_cost += 4
        elif actual == 1:
            total_cost += 15

    return total_cost
def threshold_cost(y_true, probabilities, threshold):
    flagged = probabilities >= threshold

    return order_cost(y_true, flagged)