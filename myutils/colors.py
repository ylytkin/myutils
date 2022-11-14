from matplotlib.colors import CSS4_COLORS, TABLEAU_COLORS

__all__ = [
    "ALL_COLORS",
    "COLOR_NAMES",
    "COLORS",
]

ALL_COLORS = dict(TABLEAU_COLORS)
ALL_COLORS.update(CSS4_COLORS)

# some hand-picked color names for basic plotting
COLOR_NAMES = list(TABLEAU_COLORS) + [
    "blue",
    "gold",
    "lime",
    "red",
    "magenta",
    "peru",
    "navy",
    "dodgerblue",
    "orangered",
    "mediumspringgreen",
    "rebeccapurple",
    "indianred",
    "mediumslateblue",
    "coral",
    "darkgoldenrod",
    "olivedrab",
    "palegreen",
    "darkslategray",
    "steelblue",
    "indigo",
    "mediumvioletred",
]
COLORS = [ALL_COLORS[color_name] for color_name in COLOR_NAMES]
