"""The 4 comparison approaches for next-day direction prediction.

See PLAN.md > Approaches. No hyperparameter tuning. Scaler is fit on training
data only (inside the Pipeline), never on the full dataset.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def persistence_predict(X: pd.DataFrame) -> pd.Series:
    """Predict tomorrow's direction = today's direction (Close_t > Close_{t-1}).

    `ret_1d` = Close_t / Close_{t-1} - 1 is already a same-day (no future info) feature,
    so its sign IS today's direction. Ties (ret_1d == 0) count as Down, matching the
    target's tie convention.
    """
    pred = (X["ret_1d"] > 0).astype(int)
    pred.name = "persistence_pred"
    return pred


def majority_predict(y_train: pd.Series, index: pd.Index) -> pd.Series:
    """Predict the majority class of y_train (training data only) for every row in `index`."""
    majority_class = int(y_train.mode().iloc[0])
    pred = pd.Series(majority_class, index=index, name="majority_pred")
    return pred


def make_pipeline() -> Pipeline:
    """StandardScaler + LogisticRegression, no tuning. Fit only ever on training data."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.Series:
    """Fit a fresh pipeline on (X_train, y_train), predict on X_test."""
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    pred = pd.Series(pipe.predict(X_test), index=X_test.index, name="model_pred")
    return pred, pipe
