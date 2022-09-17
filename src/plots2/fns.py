import matplotlib.pyplot as plt
import seaborn as sns
from math import floor


def get_corner_annotations(nx, ny, x0, y0, dx, dy, plt):
    letters = 'ABCDEFGHIJKLMNOP'

    for ii in range(nx*ny):
        ll = letters[ii]

        xx = x0 + dx*(ii % nx)
        yy = y0 - dy*floor(ii/nx)

        plt.annotate(
            ll,
            (xx, yy),
            xycoords='figure fraction',
            color=(0.4, 0.4, 0.4),
            size=16,
        )

    return None


def get_corner_annotations_explicit(nx, ny, xs, ys, plt):
    letters = 'ABCDEFGHIJKLMNOP'

    for ii in range(nx*ny):
        ll = letters[ii]

        xind = ii % nx
        yind = floor(ii/nx)

        xx = xs[xind]
        yy = ys[yind]

        plt.annotate(
            ll,
            (xx, yy),
            xycoords='figure fraction',
            color=(0.4, 0.4, 0.4),
            size=16,
        )

    return None


def get_corner_annotations_custom_labels(nx, ny, x0, y0, dx, dy, plt, labels):

    for ii in range(nx*ny):
        ll = labels[ii]

        xx = x0 + dx*(ii % nx)
        yy = y0 - dy*floor(ii/nx)

        plt.annotate(
            ll,
            (xx, yy),
            xycoords='figure fraction',
            color=(0.4, 0.4, 0.4),
            size=16,
        )

    return None


def get_dose_colors(N=10):
    clrmap = plt.cm.viridis_r
    cmaplist = [clrmap(floor((i / (N-1))*clrmap.N)) for i in range(N)]
    colors = sns.color_palette(cmaplist, n_colors=N)
    return colors
