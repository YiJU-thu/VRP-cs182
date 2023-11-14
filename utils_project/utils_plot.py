import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm # colormap

import numpy as np
import pandas as pd

from datetime import datetime, timedelta


def customize_plt(**kwargs):
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12  # fontsize of legend
    plt.rcParams['legend.title_fontsize'] = 12  # fontsize of legend title
    plt.rcParams['legend.frameon'] = False # remove the box of legend
    plt.rcParams['legend.handlelength'] = 2
    plt.rcParams['legend.columnspacing'] = 1
    plt.rcParams['legend.labelspacing'] = 0.25
    plt.rcParams['axes.labelsize'] = 16  # fontsize of axes title
    plt.rcParams['axes.titlesize'] = 16  # fontsize of subplot title
    plt.rcParams['xtick.labelsize'] = 14  # fontsize of ticklabels
    plt.rcParams['ytick.labelsize'] = 14  # fontsize of ticklabels
    plt.rcParams['lines.linewidth'] = 2  # width of line
    # plt.rcParams['patch.linewidth'] = 2

    for key, value in kwargs.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value
        else:
            raise ValueError(f"Invalid key: {key}")

# customize_plt()

def twin_axis(ax, axis="x", color=None, color_origin=None):
    if axis == "x":
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        if color is not None:
            ax2.spines["top"].set_color(color)
            ax2.tick_params(axis="x", colors=color)
            ax2.xaxis.label.set_color(color)
        if color_origin is not None:
            ax.spines["bottom"].set_color(color_origin)
            ax.tick_params(axis="x", colors=color_origin)
            ax.xaxis.label.set_color(color_origin)
    elif axis == "y":
        ax2 = ax.twinx()
        if color is not None:
            ax2.spines["right"].set_color(color)
            ax2.tick_params(axis="y", colors=color)
            ax2.yaxis.label.set_color(color)
        if color_origin is not None:
            ax.spines["left"].set_color(color_origin)
            ax.tick_params(axis="y", colors=color_origin)
            ax.yaxis.label.set_color(color_origin)
    else:
        raise ValueError("axis should be 'x' or 'y'")
    return ax2