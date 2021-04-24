import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

__all__ = [
    'matplotlib_latex',
    'matplotlib_svg',
    'matplotlib_seaborn_style',
    'matplotlib_dark_theme',

]

latex_rcparams = {
    'text.usetex': True,
    'text.latex.preamble': '\\usepackage[utf8]{inputenc}\n\\usepackage[russian]{babel}',
}

serif_rcparams = {
    'font.family': [
        'serif'
    ],
    'font.serif': [
        'Computer Modern Roman',
        'Times',
        'Palatino',
        'New Century Schoolbook',
        'Bookman',
    ],
}


def matplotlib_latex(serif: bool = True):
    plt.rcParams.update(latex_rcparams)

    if serif:
        plt.rcParams.update(serif_rcparams)


def matplotlib_svg():
    set_matplotlib_formats('svg')


def matplotlib_seaborn_style(palette: str = 'deep'):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette(palette))
    plt.rcParams['grid.linestyle'] = 'dotted'
    
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    
    
class matplotlib_dark_theme():
    def __init__(self, facecolor: str = 'None'):
        self.dark_theme_rcparams = self._get_dark_theme_rcparams(black=facecolor)
        self.prev_theme_rcparams = {key: plt.rcParams[key] for key in self.dark_theme_rcparams}
        
        plt.rcParams.update(self.dark_theme_rcparams)
        
    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.rcParams.update(self.prev_theme_rcparams)
        
    @staticmethod
    def _get_dark_theme_rcparams(
            white: str = '#eeeeee',
            black: str = '#111111',
            lightgray: str = 'gray',
    ) -> dict:
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


def smooth_time_series(ts: pd.Series, step: int = 6) -> pd.Series:
    ts_dense = ts.copy()

    dense_index = ts_dense.index.tolist()
    
    for i1, i2 in zip(ts_dense.index, ts_dense.index[1:]):
        delta = i2 - i1
        
        for i in range(1, step):
            new_idx = i1 + delta / step * i
            dense_index.append(new_idx)

    dense_index = sorted(dense_index)

    ts_dense = ts_dense.reindex(dense_index).reset_index(drop=True)
    
    ts_dense = ts_dense.interpolate(method='quadratic' if ts.size > 2 else 'linear')

    ts_dense.index = dense_index
    
    return ts_dense
