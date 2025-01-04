from typing import Optional, Literal, List

import numpy as np
from matplotlib import pyplot as plt


def create_horizontal_barplot(names: List[str], values: List[float], filename: str, title="Horizontal Bar Plot", xlabel="Values", ylabel="Names",
                              color="tab:blue", alphas: Optional[List[float]] = None, sort_by: Optional[Literal['abs', 'id']] = 'id',
                              max_n_values: int = -1, max_label_length: int = 50,
                              remove_nonpositive: bool = False,
                              remove_zeros: bool = False):
    """
    Creates a horizontal bar plot.

    Parameters:
        names (list of str): List of names (labels for bars).
        values (list of float): List of values (lengths of bars).
        title (str, optional): Title of the plot. Default is "Horizontal Bar Plot".
        xlabel (str, optional): Label for the x-axis. Default is "Values".
        ylabel (str, optional): Label for the y-axis. Default is "Names".
        color (str, optional): Color of the bars. Default is "blue".
    """
    # from ChatGPT
    if len(names) != len(values):
        raise ValueError("The length of names and values must be the same.")

    if sort_by is not None:
        if sort_by == 'id':
            perm = np.argsort(values)[::-1]
        elif sort_by == 'abs':
            perm = np.argsort(np.abs(np.asarray(values)))[::-1]
        else:
            raise ValueError(f'Unknown sort_by option "{sort_by}"')

        names = [names[i] for i in perm]
        values = [values[i] for i in perm]
        if alphas is not None:
            alphas = [alphas[i] for i in perm]

    if max_label_length > 0:
        for i in range(len(names)):
            if len(names[i]) > max_label_length:
                names[i] = names[i][:max_label_length] + '...'

    if remove_nonpositive:
        names = [n for n, v in zip(names, values) if v > 0]
        values = [v for n, v in zip(names, values) if v > 0]
        if alphas is not None:
            alphas = [a for a, v in zip(alphas, values) if v > 0]

    if remove_zeros:
        names = [n for n, v in zip(names, values) if v != 0]
        values = [v for n, v in zip(names, values) if v != 0]
        if alphas is not None:
            alphas = [a for a, v in zip(alphas, values) if v != 0]

    if max_n_values > 0 and len(values) > max_n_values:
        names = names[:max_n_values]
        values = values[:max_n_values]
        if alphas is not None:
            alphas = alphas[:max_n_values]

    plt.figure(figsize=(10, len(names) * 0.3))
    bars = plt.barh(names, values, color=color)
    if alphas is not None:
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(max(0.0, min(1.0, alpha)))

    if any([v < -1e-8 for v in values]):
        plt.axvline(0.0, color='k')

    plt.margins(y=0.005)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to display top-to-bottom ordering
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()
