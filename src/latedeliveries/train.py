import json
import joblib
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from latedeliveries.data import make_orders
from latedeliveries.pipeline import (
    CATEGORICAL,
    NUMERIC,
    build_preprocessor,
)


def evaluate_model(name, classifier, x_train, x_test, y_train, y_test):
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    recall = recall_score(y_test, predictions)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(f"Recall: {recall:.3f}")
    print(classification_report(y_test, predictions))


def main():
    df = make_orders()

    features = NUMERIC + CATEGORICAL

    x = df[features]
    y = df["late"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    models = {
        "Dummy": DummyClassifier(),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
        ),
    }

    print("\nMODEL COMPARISON")

    for name, classifier in models.items():
        evaluate_model(
            name,
            classifier,
            x_train,
            x_test,
            y_train,
            y_test,
        )

    print("\n" + "=" * 50)
    print("GRID SEARCH")
    print("=" * 50)

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(x_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV Recall:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(x_test)

    print("\nTest Set Results:")
    print(classification_report(y_test, predictions))

    probabilities = best_model.predict_proba(x_test)[:, 1]

    print("\n" + "=" * 50)
    print("THRESHOLD SEARCH")
    print("=" * 50)

    thresholds = np.arange(0.10, 0.95, 0.05)

    best_threshold = None
    best_cost = float("inf")
    
    for threshold in thresholds:
        flags = probabilities >= threshold

        cost = 0

        for actual, flagged in zip(y_test, flags):
            if flagged:
                cost += 4
            elif actual == 1:
                cost += 15

        print(f"Threshold={threshold:.2f} Cost={cost}")

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    print("\nBest Threshold:")
    print(best_threshold)

    print("\nBest Cost:")
    print(best_cost)


if __name__ == "__main__":
    main()

    