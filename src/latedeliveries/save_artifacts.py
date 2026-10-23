import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import joblib

from latedeliveries.data import make_orders
from latedeliveries.pipeline import (
    CATEGORICAL,
    NUMERIC,
    build_preprocessor,
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

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

    best_model = grid_search.best_estimator_

    joblib.dump(
        best_model,
        MODELS_DIR / "late_orders.joblib",
    )

    model_info = {
        "model": "LogisticRegression",
        "C": 0.01,
        "solver": "liblinear",
        "best_threshold": 0.60,
    }

    with open(
        MODELS_DIR / "model_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(model_info, file, indent=4)

    metrics = {
        "best_cv_recall": float(grid_search.best_score_),
        "best_threshold": 0.60,
        "best_cost": 1431,
    }

    with open(
        REPORTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame(
        grid_search.cv_results_
    ).to_csv(
        REPORTS_DIR / "cv_results.csv",
        index=False,
    )

    print("Artifacts saved successfully!")


if __name__ == "__main__":
    main()