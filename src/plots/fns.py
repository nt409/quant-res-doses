import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from colour import Color
import statistics
import itertools
from PIL import ImageColor

from polymodel.utils import edge_values, get_dist_mean, trait_vec, trait_vec

from plots.consts import (
    GREY_LABEL,
    LIGHT_GREY_TEXT,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    MARKER_COLOURS,
    TYPE_PLOT_MAP,
    PERCENTILES,
    KEY_ATTRS,
    FILL_OPACITY
)

from polymodel.params import PARAMS


# --------------------------------------------------------------------
# * TOC
# utility functions

# fitting plots

# single plots
# - general/compare
# - dists
# - within season
# - yield loss/difference etc
# - pathogen dist (maths)
# - other
# - variable
# - beta analysis

# multi plots

# testing

# close all fns: ctrl + K, ctrl + 0
# open all fns: ctrl + K, ctrl + J


# --------------------------------------------------------------------
# * utility functions

def standard_layout(legend_on, width=PLOT_WIDTH, height=PLOT_HEIGHT):
    return go.Layout(
        font=dict(size=16),
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


def generate_dist_mean(dist, traitvec):
    # should have been in utils originally, now is in some plotting code but
    # to maintain one version only!
    return get_dist_mean(dist, traitvec)


def title_generator(sprays, host, replace_cultivar_sev):

    rep_string = ''
    if (replace_cultivar_sev is not None and
            'Y' in replace_cultivar_sev):
        rep_string = 'Replacing cultivars; '

    changing = ''
    if len(sprays) > 1:
        changing = 'changing number of sprays'

    fixed = ''
    if len(host) > 1:
        fixed = f'{sprays[0]} sprays, '
        changing = f'varying host resistance'

    title_out = rep_string + fixed + changing

    return title_out


def _get_single_dist_area_trace(trait_vec, dist_array,
                                index, fillcolour_use, linecolour_use):
    """
    Used in '_get_multiple_dist_traces' and 'fitting_overview_plt'

    Return trace with a single distribution, with area coloured in. Consists of straight
    line and curved line, shaded in between.
    """

    traces_out = []

    # add straight line

    index_use = 0 if index is None else index

    straight_line = go.Scatter(x=[trait_vec[0], trait_vec[-1]],
                               y=[index_use, index_use],
                               mode='lines',
                               fill='tonexty',
                               fillcolor=fillcolour_use,
                               line={'width': 0},
                               )

    traces_out.append(straight_line)

    # add curved line

    if index is not None:
        length_vector = len(trait_vec)
        distribution = (index)*np.ones(length_vector) + dist_array[:, index-1]
    else:
        # if only plotting single dist, may pass in vector not array
        distribution = dist_array

    dist_line = go.Scatter(x=trait_vec,
                           y=distribution,
                           mode='lines',
                           name=f'Year {index}',
                           line_color=linecolour_use,
                           )

    traces_out.append(dist_line)

    return traces_out


def _get_multiple_dist_traces(trait_vec, dist_array, fillcolour_use=None,
                              linecolour_use=None, vertical=False,
                              step=None, selector=None):
    """
    Used for distribution plots.

    Returns distributions across number of years
    """

    num_years = dist_array.shape[1]

    if linecolour_use is None:
        linecolour_use = 'white'

    if fillcolour_use is None:
        colors = [x.hex for x in list(
            Color('green').range_to(Color('red'), num_years-1))]
    else:
        colors = [fillcolour_use for i in range(1, num_years)]

    lines_to_plot = []

    if selector is None:
        if step is not None:
            selector = [True if (i % step == 0 or i == 1)
                        else False for i in range(1, num_years)]
        else:
            selector = [True for i in range(1, num_years)]

    for i in itertools.compress(range(1, num_years), selector):
        area_out = _get_single_dist_area_trace(trait_vec, dist_array,
                                               i, colors[i-1], linecolour_use)
        lines_to_plot += area_out

    lines_to_plot.reverse()

    if vertical:
        for line in lines_to_plot:
            line['x'], line['y'] = line['y'], line['x']

    return lines_to_plot


def _generate_dist_and_traitvecs(model_output, TraitVecIn, scaleval=None):

    # to make small enough to fit between lines
    if scaleval is None:
        scaleval = 6

    # fungicide at full dose:
    # exp(-strain*conc), with conc=1
    # trait_vec_fung  = np.asarray([exp(-kk) for kk in TraitVecIn.k_vec])

    dist_arrays = {}

    TVS = dict(
        fung=TraitVecIn.k_vec,
        host=TraitVecIn.l_vec
    )

    EVS = dict(
        fung=TraitVecIn.k_edge_values,
        host=TraitVecIn.l_edge_values
    )

    trait_adjustment = dict(
        fung=0.08,
        host=3
    )

    for trait in ['fung', 'host']:
        dist_array = model_output[0][f'{trait}_dists']
        normaliser = EVS[trait][1] - EVS[trait][0]

        dist_arrays[trait] = dist_array / \
            (normaliser*scaleval*trait_adjustment[trait])

    return TVS['fung'], TVS['host'], dist_arrays['fung'], dist_arrays['host']


# single utils


def _name_difference(names):
    # should take pairs of names

    host_names = []
    fung_names = []

    for name in names:
        split_name = name.split(', ')
        fung_names.append(split_name[0])
        host_names.append(split_name[1])

    if host_names[0] == host_names[1]:
        return [host_names[0].capitalize(), fung_names[0], fung_names[1]]
    else:
        return [fung_names[0], host_names[0], host_names[1]]


def _single_trace_labels(single_run_input, sprays_plot, host_plot):

    trace_attrs = {}

    for key in single_run_input.keys():
        # generate label
        host_lab = ''
        spray_lab = ''

        if len(host_plot) > 1:
            if KEY_ATTRS[key]['host'] == 'Y':
                host_lab = 'Resistant host'
            else:
                host_lab = 'Non-resistant host'

            if len(sprays_plot) > 1:
                # also need comma
                host_lab = ', ' + host_lab.lower()

        if len(sprays_plot) > 1:
            spray_lab = 'Sprays: ' + KEY_ATTRS[key]['sprays']

        trace_attrs[key] = {
            'label': spray_lab + host_lab,
        }

    return trace_attrs


def _single_run_traces(single_run_input,
                       trait_attrs_sing,
                       sprays_plot,
                       host_plot,
                       plot,
                       replace_cultivar_sev=None,
                       cumulative=False):

    traces_to_plot = []

    for key in single_run_input.keys():
        data_list = single_run_input[key]

        if KEY_ATTRS[key]['host'] in host_plot and KEY_ATTRS[key]['sprays'] in sprays_plot:
            for data in data_list:
                if len(data) > 0:
                    y = data[TYPE_PLOT_MAP[plot]]

                    if cumulative and plot != 'DS':
                        y = np.cumsum(y)

                    xx = np.asarray(range(1, len(y)+1))

                    # Y so can replace
                    if not cumulative and replace_cultivar_sev is not None and plot != 'econ' and KEY_ATTRS[key]['host'] == 'Y':
                        if plot == 'DS':
                            index = np.argwhere(y > replace_cultivar_sev)
                        elif plot == 'yield':
                            index = np.argwhere(
                                y < PARAMS.max_yield*(1+PARAMS.yield_gradient*replace_cultivar_sev))

                        if len(index) > 0:

                            for i in range(len(index)):
                                # add i to compensate for introduction of previous 'None's
                                ind = int(i+index[i])
                                # for dotted line joining change of cultivar
                                y1 = y[ind:ind+2]
                                # for dotted line joining change of cultivar
                                xx2 = xx[ind:ind+2]

                                traces_to_plot.append(
                                    go.Scatter(
                                        x=xx2,
                                        y=y1,
                                        mode='lines',
                                        legendgroup=KEY_ATTRS[key]['sprays'],
                                        showlegend=False,
                                        line={
                                            'color': KEY_ATTRS[key]['colour'], 'dash': KEY_ATTRS[key]['dash']},
                                        marker_symbol=KEY_ATTRS[key]['symbol'],
                                        marker_size=8,
                                        name=trait_attrs_sing[key]['label']
                                    )
                                )

                                y = np.concatenate(
                                    [y[:ind+1], [None], y[ind+1:]])
                                xx = np.concatenate(
                                    [xx[:ind+1], [None], xx[ind+1:]])

                    traces_to_plot.append(
                        go.Scatter(
                            x=xx,
                            y=y,
                            mode='lines+markers',
                            legendgroup=KEY_ATTRS[key]['sprays'],
                            line={
                                'color': KEY_ATTRS[key]['colour'], 'dash': KEY_ATTRS[key]['dash']},
                            marker_symbol=KEY_ATTRS[key]['symbol'],
                            marker_size=8,
                            name=trait_attrs_sing[key]['label']
                        )
                    )

    if replace_cultivar_sev is not None and not cumulative and plot != 'econ':
        if plot == 'DS':
            yy = replace_cultivar_sev
        elif plot == 'yield':
            yy = PARAMS.max_yield * \
                (1 + PARAMS.yield_gradient*replace_cultivar_sev)
        rc = go.Scatter(
            x=[1, len(y)+1],
            y=[yy, yy],
            mode='lines',
            opacity=1,
            name='Cultivar replacement threshold',
            line={'dash': 'dash', 'color': 'green'},
        )
        traces_to_plot.append(rc)

    return traces_to_plot


def _single_run_traces_x_axis_sprays(single_run_input, trace_labels, sprays_plot,
                                     host_plot, plot, replace_cultivar_sev=None, cumulative=False):

    traces_to_plot = []

    for key in single_run_input.keys():
        data_list = single_run_input[key]
        if KEY_ATTRS[key]['host'] in host_plot and KEY_ATTRS[key]['sprays'] in sprays_plot:
            for data in data_list:
                if len(data) > 0:

                    y = data[TYPE_PLOT_MAP[plot]]

                    if cumulative and plot != 'DS':
                        y = np.cumsum(y)

                    xx = np.asarray(range(1, len(y)+1))
                    num_sprays = float(KEY_ATTRS[key]['sprays'])
                    xx = np.asarray([num_sprays*xval for xval in xx])

                    traces_to_plot.append(
                        go.Scatter(
                            x=xx,
                            y=y,
                            mode='lines+markers',
                            legendgroup=KEY_ATTRS[key]['sprays'],
                            line={
                                'color': KEY_ATTRS[key]['colour'], 'dash': KEY_ATTRS[key]['dash']},
                            marker_symbol=KEY_ATTRS[key]['symbol'],
                            marker_size=8,
                            name=trace_labels[key]['label']
                        )
                    )

    if replace_cultivar_sev is not None and not cumulative and plot != 'econ':
        if plot == 'DS':
            yy = replace_cultivar_sev
        elif plot == 'yield':
            yy = PARAMS.max_yield * \
                (1 + PARAMS.yield_gradient*replace_cultivar_sev)
        rc = go.Scatter(
            x=[1, len(y)+1],
            y=[yy, yy],
            mode='lines',
            opacity=1,
            name='Cultivar replacement threshold',
            line={'dash': 'dash', 'color': 'green'},
        )
        traces_to_plot.append(rc)

    return traces_to_plot


def _sing_run_traces_spr_x_dist_y(single_run_input,
                                  TraitVec,
                                  trace_label,
                                  sprays_plot,
                                  host_plot,
                                  plot,
                                  replace_cultivar_sev=None,
                                  cumulative=False,
                                  times_sprays=True):

    traces_to_plot = []

    for key in single_run_input.keys():
        data_list = single_run_input[key]
        if KEY_ATTRS[key]['host'] in host_plot and KEY_ATTRS[key]['sprays'] in sprays_plot:
            for data in data_list:
                if len(data) > 0:
                    dist = data['fung_dists']

                    y = generate_dist_mean(dist, TraitVec.k_vec)

                    num_sprays = float(KEY_ATTRS[key]['sprays'])

                    xx = np.asarray(range(1, len(y)+1))

                    if times_sprays:
                        y = num_sprays*y

                    if not times_sprays:
                        xx = np.asarray([num_sprays*(xval-1) for xval in xx])

                    traces_to_plot.append(
                        go.Scatter(
                            x=xx,
                            y=y,
                            mode='lines+markers',
                            legendgroup=KEY_ATTRS[key]['sprays'],
                            line={'color': KEY_ATTRS[key]['colour']},
                            opacity=1,
                            name=trace_label[key]['label']
                        )
                    )

    if replace_cultivar_sev is not None and not cumulative and plot != 'econ':
        if plot == 'DS':
            yy = replace_cultivar_sev
        elif plot == 'yield':
            yy = PARAMS.max_yield * \
                (1 + PARAMS.yield_gradient*replace_cultivar_sev)
        rc = go.Scatter(
            x=[1, len(y)+1],
            y=[yy, yy],
            mode='lines',
            opacity=1,
            name='Cultivar replacement threshold',
            line={'dash': 'dash', 'color': 'green'},
        )
        traces_to_plot.append(rc)

    return traces_to_plot


def generate_dist_var(dist, trait_vec):
    means = generate_dist_mean(dist, trait_vec)

    trait_n = dist.shape[0]
    n_years = dist.shape[1]

    vars = np.zeros(n_years)
    for yy in range(n_years):
        for dd in range(trait_n):
            vars[yy] += dist[dd, yy]*(trait_vec[dd] - means[yy])**2

    return vars


# multi utils

def _multi_trace_attrs_func(input_list, sprays_plot, host_plot, repl_plot):

    trace_attrs = {}
    colour_ind = -1
    for key in input_list.keys():

        host_res_on = key.split('host_')[1][0]  # [-1] # gives Y/N
        repl = key.split('rep=')[1][0]

        if key.startswith('spray_N'):
            spray_num = '0'
        else:
            split_string = key.split('spray_Y')
            # first element of second string is the number of sprays
            spray_num = split_string[1][0]

        violin_side = 'negative'
        if (spray_num in sprays_plot and
                host_res_on in host_plot and
                repl in repl_plot):

            colour_ind += 1
            if colour_ind > 0:
                violin_side = 'positive'

        # generate label

        # defaults
        host_lab = ''
        host_lab_short = ''
        spray_lab = ''
        repl_lab = ''

        if len(host_plot) > 1:
            if host_res_on == 'Y':
                host_lab = 'Resistant host'
                host_lab_short = 'Res. host'
            else:
                host_lab = 'Non-resistant host'
                host_lab_short = 'Non-r. host'

        if len(sprays_plot) > 1:
            spray_lab = 'Sprays: ' + spray_num
            if len(host_plot) > 1:
                spray_lab = ', ' + spray_lab

        if len(repl_plot) > 1:
            if repl == 'Y':
                repl_lab = 'Replaced cultivar'
            else:
                repl_lab = 'No replacement'

        # adds in low opacity
        fill_col = 'rgba' + \
            MARKER_COLOURS[colour_ind][3:-1] + f',{FILL_OPACITY})'

        trace_attrs[key] = {
            'fill_colour': fill_col,
            'marker_colour': MARKER_COLOURS[colour_ind],
            'label': host_lab + spray_lab + repl_lab,
            'sprays': spray_num,
            'repl': repl,
            'host_res_on': host_res_on,
            'violin_side': violin_side,
            'short_label': host_lab_short + spray_lab
        }

    return trace_attrs


def _multi_run_traces(input_list, sprays_plot, host_plot, repl_plot, plot,
                      plot_scatter_points, violin_plot, replace_cultivar_sev, cumulative):
    average_type = 'median'  # mean or median

    trace_attrs = _multi_trace_attrs_func(
        input_list, sprays_plot, host_plot, repl_plot)

    spanz = {
        'yield': [PARAMS.max_yield*(1+PARAMS.yield_gradient), PARAMS.max_yield],
        'econ': [0, PARAMS.wheat_price*PARAMS.max_yield],
        'DS': [0, 1]
    }

    traces_to_plot = []
    averages = []

    for key in input_list.keys():
        data_list = input_list[key]

        y_list = []
        if (trace_attrs[key]['host_res_on'] in host_plot and
                trace_attrs[key]['sprays'] in sprays_plot and
                trace_attrs[key]['repl'] in repl_plot):
            for data in data_list:
                if len(data) > 0 and len(data[0]) > 0:

                    # DS/yield/econ
                    y = data[0][TYPE_PLOT_MAP[plot]]

                    if cumulative and plot != 'DS':
                        y = np.cumsum(y)

                    y_list.append(y)

                    if plot_scatter_points:
                        x_vals = np.asarray(range(1, len(y)+1))
                        sdRand = 0.05
                        noise = np.random.normal(0, sdRand, len(x_vals))
                        x_vals = x_vals + noise

                        traces_to_plot.append(
                            go.Scatter(
                                x=x_vals,
                                y=y,
                                mode='markers',
                                showlegend=False,
                                marker=dict(
                                    color=trace_attrs[key]['marker_colour'],
                                    # symbol = 'x',
                                    opacity=FILL_OPACITY
                                )
                            )
                        )

            if violin_plot:
                # ticks = np.asarray(range(1,len(y)+1,2))
                # range_use = [ticks[0]-0.5,ticks[-1]+1.5]

                showledge = True
                if not cumulative:
                    spanUse = spanz[plot]

                for i in range(len(y)):
                    if i >= 1:
                        showledge = False
                    xx = [np.asarray(range(1, len(y)+1))[i]
                          for yy in range(len(y_list))]
                    y_violin = [yy[i] for yy in y_list]

                    if cumulative:
                        print(xx)  # should be all the same
                        spanUse = [xx[0]*i for i in spanz[plot]]

                    traces_to_plot.append(
                        go.Violin(
                            x=xx,
                            y=y_violin,
                            meanline_visible=False,
                            points=False,
                            span=spanUse,
                            scalemode='count',
                            showlegend=showledge,
                            legendgroup=trace_attrs[key]['host_res_on'],
                            scalegroup=trace_attrs[key]['label'],
                            name=trace_attrs[key]['label'],
                            side=trace_attrs[key]['violin_side'],
                            opacity=0.4,
                            line_color=trace_attrs[key]['marker_colour'])
                    )

            if len(y_list) > 0:
                if average_type == 'mean':
                    y_average = np.asarray(
                        [statistics.mean([yy[yr] for yy in y_list]) for yr in range(len(y))])
                else:
                    y_average = np.asarray(
                        [statistics.median([yy[yr] for yy in y_list]) for yr in range(len(y))])

                percentiles_list = []
                for percentile_number in PERCENTILES:
                    percentiles_list.append(np.asarray([np.percentile(
                        [yy[yr] for yy in y_list], percentile_number) for yr in range(len(y))]))

                if not violin_plot:
                    fill_list = ['tonexty']*len(percentiles_list)
                    fill_list[0] = None

                    for yy, fill in zip(percentiles_list, fill_list):
                        traces_to_plot.append(
                            go.Scatter(
                                x=np.asarray(range(1, len(yy)+1)),
                                y=yy,
                                mode='lines',
                                opacity=1,
                                name=trace_attrs[key]['label'],
                                showlegend=False,
                                fill=fill,
                                fillcolor=trace_attrs[key]['fill_colour'],
                                line={'width': 0},
                            )
                        )

                if not violin_plot:
                    # + ': ' + average_type
                    av_label = trace_attrs[key]['label']
                else:
                    # + ': ' + average_type
                    av_label = trace_attrs[key]['short_label']

                averages.append(
                    go.Scatter(
                        x=np.asarray(range(1, len(y_average)+1)),
                        y=y_average,
                        mode='markers+lines',
                        legendgroup=trace_attrs[key]['host_res_on'],
                        line={'color': trace_attrs[key]['marker_colour']},
                        opacity=1,
                        name=av_label
                    )
                )

    traces_to_plot.extend(averages)

    if replace_cultivar_sev is not None and not cumulative and plot != 'econ':
        if plot == 'DS':
            yy = replace_cultivar_sev
        elif plot == 'yield':
            yy = PARAMS.max_yield * \
                (1 + PARAMS.yield_gradient*replace_cultivar_sev)
        rc = go.Scatter(
            x=[1, len(y_average)+1],
            y=[yy, yy],
            mode='lines',
            opacity=1,
            name='Cultivar replacement threshold',
            # showlegend= False,
            # fillcolor = 'black',
            line={'dash': 'dash', 'color': 'green'},
        )
        traces_to_plot.append(rc)
    return traces_to_plot


def _difference_traces(y0, y1):
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)

    differences = y1-y0

    percentiles_list = []
    for percentile_number in PERCENTILES:
        percentiles_list.append(np.asarray([np.percentile(
            [differences[:, yr]], percentile_number) for yr in range(differences.shape[1])]))

    fill_list = ['tonexty']*len(percentiles_list)
    fill_list[0] = None

    percentile_traces = []

    for yy, fill in zip(percentiles_list, fill_list):

        percentile_traces.append(
            go.Scatter(
                x=np.asarray(range(1, len(yy)+1)),
                y=yy,
                mode='lines',
                opacity=1,
                showlegend=False,
                fill=fill,
                fillcolor=f'rgba(0,255,0,{FILL_OPACITY})',
                line={'width': 0},
            )
        )

    av_diff = np.asarray([statistics.median(differences[:, yr])
                         for yr in range(differences.shape[1])])

    median_trace = go.Scatter(
        x=np.asarray(range(1, len(av_diff)+1)),
        y=av_diff,
        mode='markers+lines',
        legendgroup='diffs',
        showlegend=False,
        line={'color': 'rgb(0,255,0)'},
        opacity=1,
    )

    return percentile_traces, median_trace


def _multi_run_differences_traces(input_list, sprays_plot, host_plot, repl_plot, plot):

    trace_attrs = _multi_trace_attrs_func(
        input_list, sprays_plot, host_plot, repl_plot)

    traces_to_plot = []

    y_list = {}
    keys_used = []
    sprays_label = []
    hosts_label = []

    for key in input_list.keys():
        data_list = input_list[key]

        if (trace_attrs[key]['host_res_on'] in host_plot and
                trace_attrs[key]['sprays'] in sprays_plot and
                trace_attrs[key]['repl'] in repl_plot):

            y_list[key] = []
            for data in data_list:

                if len(keys_used) == 0 or key not in keys_used:
                    keys_used.append(key)
                    sprays_label.append(trace_attrs[key]['sprays'])
                    hosts_label.append(trace_attrs[key]['short_label'])

                if len(data) > 0 and len(data[0]) > 0:
                    y = data[0][TYPE_PLOT_MAP[plot]]

                    y_list[key].append(y)

    y0 = y_list[keys_used[0]]
    y1 = y_list[keys_used[1]]

    percentile_traces, median_trace = _difference_traces(y0, y1)

    if len(sprays_plot) > 1:
        name_use = f'Difference: {sprays_label[1]} sprays - {sprays_label[0]} sprays'
    else:
        name_use = f'Difference: {hosts_label[1]} - {hosts_label[0]}'

    median_trace['name'] = name_use
    median_trace['showlegend'] = True

    traces_to_plot = traces_to_plot + percentile_traces
    traces_to_plot.append(median_trace)

    return traces_to_plot


# * end of utility functions


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
    >>>dists = generate_dists(data_single['spray_Y2_host_N'], conf_sing, 2)
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
        e.g. multi_output['spray_Y2_host_N']

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
        e.g. multi_output['spray_Y2_host_N']

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

        yy = generate_dist_mean(
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


def traces_with_uncertainty_bands(df,
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


def mutual_difference_traces(mutual_sevs, bds, color):

    run_1 = list(mutual_sevs.keys())[0].split('without')[0]
    run_2 = list(mutual_sevs.keys())[-1].split('without')[0]

    trcs = []

    for trait in ['without_host', 'without_fung']:

        y1 = np.asarray(mutual_sevs[run_1 + trait])
        y2 = np.asarray(mutual_sevs[run_2 + trait])

        yd = y2 - y1

        df = pd.DataFrame(yd).T
        df.columns = ['y']*df.shape[1]
        df.index = df.index+1

        traces_to_plot = traces_with_uncertainty(
            df,
            bds=bds,
            color=color,
        )

        trcs.append(traces_to_plot)

    without_host = trcs[0]
    without_fung = trcs[1]

    return without_host, without_fung


def name_from_keys(key1, key2, which):

    if which == '1':
        key_use = key1
    else:
        key_use = key2

    if key1.split('spray_')[1][:2] == key2.split('spray_')[1][:2]:
        if key_use.split('host_')[1][0] == 'N':
            return 'Host on'
        else:
            return 'Host off'
    else:
        return f"Sprays {key_use.split('spray_Y')[1][0]}"


def hex_to_rgb(string):
    args = ImageColor.getcolor(string, 'RGB')
    return f'rgba({args[0]},{args[1]},{args[2]},1)'
