from typing import Dict

import pandas as pd
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats

import matplotlib.pyplot as plt

__all__ = [
    "matplotlib_latex",
    "matplotlib_svg",
    "matplotlib_seaborn_style",
    "matplotlib_dark_theme",
]

latex_rcparams = {
    "text.usetex": True,
    "text.latex.preamble": "\\usepackage[utf8]{inputenc}\n\\usepackage[russian]{babel}",
}

serif_rcparams = {
    "font.family": ["serif"],
    "font.serif": [
        "Computer Modern Roman",
        "Times",
        "Palatino",
        "New Century Schoolbook",
        "Bookman",
    ],
}


def matplotlib_latex(serif: bool = True) -> None:
    plt.rcParams.update(latex_rcparams)

    if serif:
        plt.rcParams.update(serif_rcparams)


def matplotlib_svg() -> None:
    set_matplotlib_formats("svg")


def matplotlib_seaborn_style(palette: str = "deep") -> None:
    plt.style.use("seaborn-whitegrid")
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=sns.color_palette(palette))
    plt.rcParams["grid.linestyle"] = "dotted"

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False


class matplotlib_dark_theme:  # pylint: disable=invalid-name
    def __init__(self, facecolor: str = "#111111") -> None:
        self.dark_theme_rcparams = self._get_dark_theme_rcparams(black=facecolor)
        self.prev_theme_rcparams = {key: plt.rcParams[key] for key in self.dark_theme_rcparams}

        plt.rcParams.update(self.dark_theme_rcparams)

    def __enter__(self):  # type: ignore
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        plt.rcParams.update(self.prev_theme_rcparams)

    @staticmethod
    def _get_dark_theme_rcparams(
        white: str = "#eeeeee",
        black: str = "#111111",
        lightgray: str = "gray",
    ) -> Dict[str, str]:
        return {
            "lines.color": white,
            "patch.edgecolor": white,
            "text.color": white,
            "axes.facecolor": black,
            "axes.edgecolor": lightgray,
            "axes.labelcolor": white,
            "axes.titlecolor": white,
            "xtick.color": white,
            "ytick.color": white,
            "grid.color": lightgray,
            "figure.facecolor": black,
            "figure.edgecolor": black,
            "savefig.facecolor": black,
            "savefig.edgecolor": black,
        }


def smooth_time_series(time_series: pd.Series, step: int = 6) -> pd.Series:
    time_series_dense = time_series.copy()

    dense_index = time_series_dense.index.tolist()

    for i_1, i_2 in zip(time_series_dense.index, time_series_dense.index[1:]):
        delta = i_2 - i_1

        for i in range(1, step):
            new_idx = i_1 + delta / step * i
            dense_index.append(new_idx)

    dense_index = sorted(dense_index)

    time_series_dense = time_series_dense.reindex(dense_index).reset_index(drop=True)

    time_series_dense = time_series_dense.interpolate(
        method="quadratic" if time_series.size > 2 else "linear"
    )

    time_series_dense.index = dense_index

    return time_series_dense
