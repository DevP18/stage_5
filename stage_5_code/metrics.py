from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate(
    y_true,
    y_pred
):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    prec = precision_score(
        y_true,
        y_pred,
        average="macro"
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return (
        acc,
        prec,
        rec,
        f1
    )