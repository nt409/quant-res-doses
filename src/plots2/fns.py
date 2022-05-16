import copy
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageColor

from poly2.utils import edge_values, get_dist_mean, trait_vec

from plots2.consts import (
    GREY_LABEL,
    PLOT_HEIGHT,
    PLOT_WIDTH,
)

# --------------------------------------------------------------------
# * utility functions


def standard_layout(legend_on, width=PLOT_WIDTH, height=PLOT_HEIGHT):
    return go.Layout(
        font=dict(size=14),
        template="simple_white",
        width=width,
        height=height,
        showlegend=legend_on,
        xaxis=dict(showgrid=False),
    )


def get_text_annotation(
    x,
    y,
    text,
    xanchor=None,
    yanchor=None,
    size=14,
    color=None,
    refs=None
):

    xanchor = "center" if xanchor is None else xanchor
    yanchor = "top" if yanchor is None else yanchor

    color = GREY_LABEL if color is None else color

    xref = 'paper' if refs is None else f'x{refs}'
    yref = 'paper' if refs is None else f'y{refs}'

    return dict(
        x=x,
        y=y,
        text=text,

        showarrow=False,

        xref=xref,
        yref=yref,

        xanchor=xanchor,
        yanchor=yanchor,

        font=dict(
            size=size,
            color=color,
        ),
    )


def get_arrow_annotation(
        x,
        y,
        text,
        dx,
        dy,
        size=14,
        color=None,
        refs=None,
        xanchor=None,
        yanchor=None):

    color = GREY_LABEL if color is None else color

    xref = 'paper' if refs is None else f'x{refs}'
    yref = 'paper' if refs is None else f'y{refs}'

    xanch = 'center' if xanchor is None else xanchor
    yanch = 'top' if yanchor is None else yanchor

    return dict(
        x=x,
        y=y,

        text=text,

        showarrow=True,
        arrowcolor=color,
        arrowsize=1,
        arrowwidth=1,
        arrowhead=2,

        ax=dx,
        ay=dy,

        xref=xref,
        yref=yref,

        xanchor=xanch,
        yanchor=yanch,

        font=dict(
            size=size,
            color=color,
        ),
    )


# * FEB 02/2022

def generate_dists(data, config, scale_val=1):
    """Scaled pathogen distributions

    Parameters
    ----------
    data : dict
        model output - single[key]
        or multi[key][0]

    config : Config

    scale_val : float, optional
        scaling if want bigger dists

    Returns
    -------
    dist_arrays : dict
        keys: 
        - 'host'
        - 'fung'

    Example
    -------
    >>>dists = generate_dists(data_single['spray_2_host_N'], conf_sing, 2)
    """
    EVS = {
        'host': edge_values(config.n_l),
        'fung': edge_values(config.n_k),
    }

    SCALING = dict(
        fung=0.48,
        host=18
    )

    dist_arrays = {}

    for trait in ['fung', 'host']:
        dist_array = data[f'{trait}_dists']
        normaliser = EVS[trait][1] - EVS[trait][0]

        dist_arrays[trait] = (
            scale_val * dist_array / (normaliser * SCALING[trait])
        )

    return dist_arrays


def get_dist_traces(
        dist_arrays,
        trait,
        colour,
        years_step=1,
        fill_opacity=0.1,
        line_opacity=0.5,
):
    """Traces for a single trait - distributions over n years

    Parameters
    ----------
    dist_arrays : dict
        output of generate_dists
    trait : str
        'fung' or 'host'
    colour : str
        e.g. rgba(0,0,255,0)
    years_step : int, optional
        e.g. if want every second year pick 2

    Returns
    -------
    _type_
        _description_

    Example
    -------
    >>>trcs = get_dist_traces(dist_arrays, 'fung', 'rgba(0,0,255,1)', 5)
    """

    trcs = []

    n_trait = len(dist_arrays[trait][:, 0])

    if trait == 'host':
        EVs = trait_vec(n_trait)
    else:
        EVs = trait_vec(n_trait)

    n_years = len(dist_arrays[trait][0, :])

    for ii in range(0, n_years, years_step):
        year = n_years - ii - 1

        dist_this_year = dist_arrays[trait][:, year]

        trc = go.Scatter(
            x=year + dist_this_year,
            y=EVs,
            line=dict(
                width=1,
                color=colour[:-2] + f'{line_opacity})'
            ),
            showlegend=False,
            mode='lines',
        )

        trcs.append(trc)

        str8 = go.Scatter(
            x=[year, year],
            y=[EVs[0], EVs[-1]],
            fill='tonexty',
            fillcolor=colour[:-2] + f'{fill_opacity})',
            line={'width': 0, 'color': colour},
            showlegend=False,
            mode='lines',
            # name='straight',
        )

        trcs.append(str8)

    return trcs


def transpose_traces(trcs):
    """Mirror image x and y for list of traces"""

    new_trcs = []

    for trc in trcs:
        tmp = copy.copy(trc)
        tmp['x'] = trc['y']
        tmp['y'] = trc['x']

        new_trcs.append(tmp)

    return new_trcs


def arrange_as_data_frame(data, variable):
    """For use with traces with uncertainty

    Parameters
    ----------
    data : list
        e.g. multi_output['spray_2_host_N']

    variable : str
        e.g. 'dis_sev'

    Returns
    -------
    combined : pd.DataFrame
        M x N with:
        - N columns with label 'y'
        - index year 1,...,M

    """
    combined = pd.DataFrame()

    for ii in range(len(data)):
        tmp = (
            pd.DataFrame({'y': data[ii][variable]})
            .assign(year=lambda df: df.index + 1)
            .set_index('year')
        )

        combined = pd.concat([combined, tmp], axis=1)

    return combined


def dist_means_as_df(data, variable, conf):
    """_summary_

    Parameters
    ----------
    data : list
        e.g. multi_output['spray_2_host_N']

    variable : str
        'host' or 'fung'

    Returns
    -------
    combined : pd.DataFrame
        index year, n_iterations columns with name y
    """
    combined = pd.DataFrame()

    for ii in range(len(data)):

        if variable == 'host':
            tv = trait_vec(conf.n_l)
        else:
            tv = trait_vec(conf.n_k)

        yy = get_dist_mean(
            data[ii][f'{variable}_dists'],
            tv
        )

        tmp = (
            pd.DataFrame({'y': yy})
            .assign(year=lambda df: df.index + 1)
            .set_index('year')
        )

        combined = pd.concat([combined, tmp], axis=1)

    return combined


def traces_with_uncertainty(df,
                            bds=[25, 75],
                            color=None,
                            dash=None,
                            name=None,
                            showlegend=False
                            ):
    """Mean and range

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, e.g. output of arrange_as_data_frame(data, variable)
        index=year, n_iterations * columns with name 'y', or y1, y2, etc
        ... just needs to be able to filter(like='y')

    variable : str
        e.g. 'yield_vec'

    bds : list
        e.g. [2.5, 97.5], default is [25,75]

    color : str
        --

    dash : str
        --

    name : str

    showlegend : bool


    Returns
    -------
    traces : list
        list of traces
    """

    out = (
        df
        .assign(
            pcl=lambda df: df.filter(like='y').apply(
                np.percentile, axis=1, q=bds[0]),
            pcu=lambda df: df.filter(like='y').apply(
                np.percentile, axis=1, q=bds[1]),
            # pcm = lambda df: df.apply(np.mean, axis=1),
            pcm=lambda df: df.filter(like='y').apply(np.median, axis=1),
        )
    )

    OUTSIDE_LINE_OPACITY = 0.4
    FILL_COLOR_OPACITY = 0.1

    THIN_LINE_WIDTH = 0.5
    THICK_LINE_WIDTH = 1.5

    trc_l = go.Scatter(
        x=out.index,
        y=out.pcl,
        showlegend=False,
        line=dict(
            color=color[:-2] + f'{OUTSIDE_LINE_OPACITY})',
            # dash='dash',
            width=THIN_LINE_WIDTH,
        ),
        mode='lines',
    )

    trc_m = go.Scatter(
        x=out.index,
        y=out.pcm,
        name=name,
        showlegend=showlegend,
        line=dict(
            color=color,
            dash=dash,
            width=THICK_LINE_WIDTH,
        ),
        mode='lines'
    )

    trc_u = go.Scatter(
        x=out.index,
        y=out.pcu,
        showlegend=False,
        line=dict(
            color=color[:-2] + f'{OUTSIDE_LINE_OPACITY})',
            # dash='dash',
            width=THIN_LINE_WIDTH,
        ),
        mode='lines',
        fill='tonexty',
        fillcolor=color[:-2] + f'{FILL_COLOR_OPACITY})',
    )

    traces = [trc_l, trc_u, trc_m]

    return traces


def traces_with_uncertainty_bands(
    df,
    bds=[5, 25, 75, 95],
    color=None,
    dash=None,
    name=None,
    showlegend=False
):
    """Mean and range

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, e.g. output of arrange_as_data_frame(data, variable)
        index=year, n_iterations * columns with name 'y'

    variable : str
        e.g. 'yield_vec'

    bds : list
        e.g. [5, 25, 75, 95]

    color : str
        --

    dash : str
        --

    name : str
        --

    showlegend : bool
        --


    Returns
    -------
    traces : list
        list of traces
    """

    out = (
        df
        .assign(
            pcl2=lambda df: df.filter(like='y').apply(
                np.percentile, axis=1, q=bds[0]),
            pcl=lambda df: df.filter(like='y').apply(
                np.percentile, axis=1, q=bds[1]),
            pcu=lambda df: df.filter(like='y').apply(
                np.percentile, axis=1, q=bds[2]),
            pcu2=lambda df: df.filter(like='y').apply(
                np.percentile, axis=1, q=bds[3]),
            pcm=lambda df: df.filter(like='y').apply(np.median, axis=1),
        )
    )

    OUTSIDE_LINE_OPACITY = 0.4
    FILL_COLOR_OPACITY = 0.1
    FILL_COLOR_OPACITY2 = 0.2

    V_THIN_LINE_WIDTH = 0.5
    THIN_LINE_WIDTH = 0.5
    THICK_LINE_WIDTH = 1.5

    trc_l2 = go.Scatter(
        x=out.index,
        y=out.pcl2,
        showlegend=False,
        line=dict(
            color=color[:-2] + f'{OUTSIDE_LINE_OPACITY})',
            # dash='dash',
            width=THIN_LINE_WIDTH,
        ),
        mode='lines',
    )

    trc_l = go.Scatter(
        x=out.index,
        y=out.pcl,
        showlegend=False,
        line=dict(
            color=color[:-2] + f'{OUTSIDE_LINE_OPACITY})',
            width=V_THIN_LINE_WIDTH,
        ),
        mode='lines',
        fill='tonexty',
        fillcolor=color[:-2] + f'{FILL_COLOR_OPACITY})',
    )

    trc_m = go.Scatter(
        x=out.index,
        y=out.pcm,
        name=name,
        showlegend=showlegend,
        line=dict(
            color=color,
            dash=dash,
            width=THICK_LINE_WIDTH,
        ),
        mode='lines'
    )

    trc_u = go.Scatter(
        x=out.index,
        y=out.pcu,
        showlegend=False,
        line=dict(
            color=color[:-2] + f'{OUTSIDE_LINE_OPACITY})',
            width=V_THIN_LINE_WIDTH,
        ),
        mode='lines',
        fill='tonexty',
        fillcolor=color[:-2] + f'{FILL_COLOR_OPACITY2})',
    )

    trc_u2 = go.Scatter(
        x=out.index,
        y=out.pcu2,
        showlegend=False,
        line=dict(
            color=color[:-2] + f'{OUTSIDE_LINE_OPACITY})',
            width=THIN_LINE_WIDTH,
        ),
        mode='lines',
        fill='tonexty',
        fillcolor=color[:-2] + f'{FILL_COLOR_OPACITY})',
    )

    traces = [trc_l2, trc_l, trc_u, trc_u2, trc_m]

    return traces


def hex_to_rgb(string):
    args = ImageColor.getcolor(string, 'RGB')
    return f'rgba({args[0]},{args[1]},{args[2]},1)'


def corner_annotations_rowwise(
    row_n, col_n, top, left, row_gap, col_gap
):

    corners = []

    col_vec = get_cols(left, col_gap, col_n)
    row_vec = get_rows(top, row_gap, row_n)

    letters = 'ABCDEFGHIJKLMN'

    ii = 0

    for col, row in itertools.product(row_vec, col_vec):
        letter = letters[ii]
        corner = get_text_annotation(row, col, letter, size=20)
        corners.append(corner)
        ii += 1

    return corners


def corner_annotations_colwise(
    row_n, col_n, top, left, row_gap, col_gap
):
    corners = []

    col_vec = get_cols(left, col_gap, col_n)
    row_vec = get_rows(top, row_gap, row_n)

    letters = 'ABCDEFGHIJKLMN'

    ii = 0

    for row, col in itertools.product(col_vec, row_vec):
        letter = letters[ii]
        corner = get_text_annotation(row, col, letter, size=20)
        corners.append(corner)
        ii += 1

    return corners


def get_rows(top, row_gap, row_n):
    rows = [top - i*row_gap for i in range(row_n)]
    return rows


def get_cols(left, col_gap, col_n):
    cols = [left + i*col_gap for i in range(col_n)]
    return cols
