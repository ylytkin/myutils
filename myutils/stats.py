from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

__all__ = [
    "corrcoef",
    "acfunc",
]


def cluster_features(corr: NDArray[Any]) -> NDArray[Any]:
    if corr.shape == (1, 1):
        return np.array([1])

    dist = pdist(corr)

    clusters: NDArray[Any] = fcluster(
        linkage(
            dist,
            method="complete",
        ),
        t=0.5 * np.max(dist),
        criterion="distance",
    )

    return clusters


def reorder_corr(corr: pd.DataFrame) -> pd.DataFrame:
    corr = corr.copy()

    idx = corr.index.values
    corr = corr.values

    outer_clusters = cluster_features(corr)
    inner_clusters = []

    for cluster in np.unique(outer_clusters):
        mask = outer_clusters == cluster
        inner_corr = corr[mask][:, mask]

        current_inner_clusters = cluster_features(inner_corr)
        inner_clusters.append(current_inner_clusters)

    inner_clusters = np.concatenate(inner_clusters)

    clusters = pd.DataFrame(list(zip(outer_clusters, inner_clusters))).sort_values(by=[0, 1])

    reordering = clusters.index.values

    corr = pd.DataFrame(
        corr[reordering][:, reordering],
        index=idx[reordering],
        columns=idx[reordering],
    )

    return corr


def corrcoef(
    data: pd.DataFrame,
    method: str = "pearson",
    reorder: bool = True,
    alpha: Optional[float] = 0.05,
    min_periods: int = 5,
) -> pd.DataFrame:
    data = data.copy()

    corr = data.corr(method=method)

    if reorder is True:
        corr = reorder_corr(corr)
        data = data.loc[:, corr.columns]

    if alpha is not None:
        nval = 1 - np.isnan(data.values).astype(int)
        nval = nval.T.dot(nval)

        quantile = stats.t.ppf(1 - alpha / 2, df=nval - 2)
        critical = quantile / np.sqrt(nval - 2 + quantile**2)

        corr.values[np.abs(corr.values) < critical] = np.nan
        corr.values[nval < min_periods] = np.nan

    return corr


def acfunc(
    time_series: pd.Series,
    minlag: int = 2,
    maxlag: int = 14,
    method: str = "pearson",
    min_periods: int = 5,
) -> pd.Series:
    time_series_lags = pd.DataFrame({i: time_series.shift(-i) for i in range(minlag, maxlag + 1)})

    acf = time_series_lags.corrwith(time_series, method=method)
    acf.index.name = "lag"

    nval_time_series = 1 - np.isnan(time_series.values).astype(int)
    nval_lags = 1 - np.isnan(time_series_lags.values).astype(int)
    nval = nval_lags.T.dot(nval_time_series)

    acf.values[nval < min_periods] = np.nan

    return acf
