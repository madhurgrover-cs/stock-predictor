import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def persistence_predict(X: pd.DataFrame) -> pd.Series:
    pred = (X["ret_1d"] > 0).astype(int)
    pred.name = "persistence_pred"
    return pred


def majority_predict(y_train: pd.Series, index: pd.Index) -> pd.Series:
    majority_class = int(y_train.mode().iloc[0])
    pred = pd.Series(majority_class, index=index, name="majority_pred")
    return pred


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def make_rf_pipeline() -> Pipeline:
    """RandomForest doesn't need scaling, but kept in a Pipeline for symmetry
    with the other approaches. No hyperparameter tuning."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    pred = pd.Series(pipe.predict(X_test), index=X_test.index, name="model_pred")
    return pred, pipe


def fit_predict_rf(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    pipe = make_rf_pipeline()
    pipe.fit(X_train, y_train)
    pred = pd.Series(pipe.predict(X_test), index=X_test.index, name="rf_pred")
    return pred, pipe

