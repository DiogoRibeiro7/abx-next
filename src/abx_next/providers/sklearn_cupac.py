"""Sklearn-backed covariate provider suitable for CUPAC adjustments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..utils.types import CovariateProvider


class _Regressor(Protocol):
    """Protocol describing the minimal regressor surface we rely on."""

    def predict(self, features: Any) -> Any:
        """Return numeric predictions for the supplied design matrix."""


def _validate_feature_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Ensure the feature DataFrame contains the required numeric columns."""
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Feature DataFrame missing columns: {missing}")

    for name in columns:
        if not is_numeric_dtype(df[name]):
            raise TypeError(f"Feature column '{name}' must be numeric.")
        if df[name].isna().any():
            raise ValueError(f"Feature column '{name}' contains NaN values.")


@dataclass(frozen=True)
class SklearnCovariateProvider(CovariateProvider):
    """
    Generate CUPAC covariates using a pre-trained scikit-learn model.

    Parameters
    ----------
    model:
        Pre-trained regressor exposing a ``predict`` method that accepts a
        2-d numeric array and returns a 1-d array of floats.
    feature_df:
        DataFrame containing at least ``key_col`` and ``feature_cols`` needed for
        generating covariates.
    key_col:
        Column in ``feature_df`` that uniquely identifies users.
    feature_cols:
        Ordered list of numeric feature columns used during model training.
    """

    model: _Regressor
    feature_df: pd.DataFrame
    key_col: str
    feature_cols: Sequence[str]
    _feature_matrix: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not hasattr(self.model, "predict"):
            raise TypeError("model must expose a callable 'predict' attribute.")
        if not callable(self.model.predict):
            raise TypeError("'predict' attribute on model must be callable.")

        if not isinstance(self.feature_df, pd.DataFrame):
            raise TypeError("feature_df must be a pandas DataFrame.")

        if self.key_col not in self.feature_df.columns:
            raise ValueError(f"Feature DataFrame missing key column '{self.key_col}'.")

        if not self.feature_cols:
            raise ValueError("feature_cols must contain at least one column name.")

        feature_cols = list(self.feature_cols)

        if len(set(feature_cols)) != len(feature_cols):
            raise ValueError("feature_cols must not contain duplicates.")

        key_series = self.feature_df[self.key_col]
        if key_series.isna().any():
            raise ValueError(f"Key column '{self.key_col}' contains NaN values.")
        if key_series.duplicated().any():
            raise ValueError(f"Key column '{self.key_col}' must contain unique identifiers.")

        object.__setattr__(self, "feature_cols", tuple(feature_cols))

        _validate_feature_columns(self.feature_df, self.feature_cols)

        object.__setattr__(
            self,
            "_feature_matrix",
            self.feature_df.set_index(self.key_col)[list(self.feature_cols)].copy(),
        )

    def get_covariate(self, user_ids: pd.Series) -> pd.Series:
        """Return model predictions aligned to the supplied user identifiers."""
        if not isinstance(user_ids, pd.Series):
            raise TypeError("user_ids must be provided as a pandas Series.")
        if user_ids.isna().any():
            raise ValueError("user_ids must not contain NaN values.")

        index = pd.Index(user_ids)
        missing = index.difference(self._feature_matrix.index)
        if not missing.empty:
            raise ValueError(f"Missing features for user ids: {list(missing)}")

        # Reindex preserves the order (and duplicates) present in user_ids.
        features = self._feature_matrix.reindex(index.values)
        if features.isna().any().any():
            raise ValueError("Encountered NaNs after reindexing feature matrix.")

        predictions = self._predict(features)
        if predictions.shape[0] != len(user_ids):
            raise ValueError("Model returned predictions with unexpected shape.")
        if np.any(~np.isfinite(predictions)):
            raise ValueError("Model predictions must be finite numeric values.")

        return pd.Series(predictions.astype(float), index=user_ids, name="covariate")

    def _predict(self, features: pd.DataFrame) -> np.ndarray:
        """Call model.predict with the feature matrix."""
        try:
            result = self.model.predict(features)
        except TypeError:
            # Fallback for estimators that only accept numpy arrays.
            as_array = features.to_numpy(dtype=float, copy=False)
            result = self.model.predict(as_array)
        predictions = np.asarray(result, dtype=float)
        if predictions.ndim == 1:
            return predictions
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions[:, 0]
        raise ValueError("Model.predict must return a 1-dimensional array.")
