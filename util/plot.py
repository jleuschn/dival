# -*- coding: utf-8 -*-
"""Provides utility functions for visualization."""
from math import ceil
import matplotlib.pyplot as plt
import numpy as np


def plot_image(x, fig=None, ax=None, **kwargs):
    """Plot `x` using matplotlib's `imshow` method.

    Parameters
    ----------
    x : array-like or PIL image
        The image data. For further information see `imshow documentation
        <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`_.
    fig : matplotlib.figure.Figure, optional
        The figure to plot the image in. If ``fig is None``, but `ax` is given,
        it is retrieved from `ax`. If both ``fig is None`` and ``ax is None``,
        a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the image in. If ``None``, an axes object is created
        in `fig`.
    """
    if fig is None:
        if ax is None:
            fig = plt.figure()
        else:
            fig = ax.get_figure()
    if ax is None:
        ax = fig.add_subplot(111)
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('cmap', 'gray')
    xticks = kwargs.pop('xticks', None)
    yticks = kwargs.pop('yticks', None)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.imshow(x.asarray().T, **kwargs)
    return ax


def plot_images(x_list, nrows=1, ncols=-1, fig=None, **kwargs):
    """Plot multiple images using matplotlib's `imshow` method in subplots.

    Parameters
    ----------
    x_list : list of (array-like or PIL image)
        List of the image data. For further information see `imshow
        documentation
        <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`_.
    nrows : int, optional
        The number of subplot rows (the default is 1). If -1, it is computed by
        ``ceil(len(x_list)/ncols)``, or set to 1 if `ncols` is not given.
    ncols : int, optional
        The number of subplot columns. If -1, it is computed by
        ``ceil(len(x_list)/nrows)`` (default). If both `nrows` and `ncols` are
        given, the value of `ncols` is ignored.
    fig : matplotlib.figure.Figure, optional
        The figure to plot the images in. If ``None``, a new figure is created.
    """
    if fig is None:
        fig = plt.figure()
    if nrows is None or nrows == -1:
        if ncols is None or ncols == -1:
            nrows = 1
        else:
            nrows = ceil(len(x_list)/ncols)
    ncols = ceil(len(x_list)/nrows)
    ax = fig.subplots(nrows, ncols)
    if nrows == 1 and ncols == 1:
        ax = np.array(ax)
    for x, a in zip(x_list, ax.flat):
        plot_image(x, ax=a, **kwargs)
