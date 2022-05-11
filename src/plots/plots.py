"""All figures.

- within_season
- compare_strategies_overview


"""

import itertools
import statistics
from math import exp, log, log10
from turtle import bgcolor

import numpy as np
import plotly.graph_objects as go
from colour import Color
from plotly.subplots import make_subplots
from polymodel.params import PARAMS

from plots.consts import (FILL_OPACITY, FUNG_COLOUR, GREY_LABEL, HOST_COLOUR,
                          KEY_ATTRS, KEY_TO_LONGNAME, KEY_TO_SHORTNAME,
                          MARKER_COLOURS, PERCENTILES, TRAIT_FULLNAME,
                          TYPE_PLOT_MAP, Y_LABEL_MAP)

from plots.fns import (_difference_traces, _generate_dist_and_traitvecs,
                       _get_multiple_dist_traces, _get_single_dist_area_trace,
                       _multi_run_differences_traces, _multi_run_traces,
                       _name_difference, _sing_run_traces_spr_x_dist_y,
                       _single_run_traces, _single_run_traces_x_axis_sprays,
                       _single_trace_labels, arrange_as_data_frame,
                       dist_means_as_df, generate_dist_mean, generate_dist_var,
                       generate_dists, get_arrow_annotation, get_dist_traces,
                       get_text_annotation, mutual_difference_traces,
                       name_from_keys, standard_layout,
                       traces_with_uncertainty)


def dist_plot(data, config):
    col1 = 'rgba(0,100,20,1)'
    # col2 = 'rgba(50,50,50,1)'

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
    )

    fill_opacity = 0.4

    dists_f1 = generate_dists(data, config, 3)
    trcs_fd1 = get_dist_traces(dists_f1, 'fung', col1, 1, fill_opacity, 1)

    dists_f2 = generate_dists(data, config, 1.9)
    trcs_fd2 = get_dist_traces(dists_f2, 'host', col1, 1, fill_opacity, 1)

    fig.add_traces(trcs_fd1, rows=1, cols=1)

    fig.add_traces(trcs_fd2, rows=2, cols=1)

    #
    #
    # LAYOUT
    fig.update_layout(standard_layout(True, height=800))
    fig.update_layout(font=dict(size=14))

    fig.update_xaxes(title_text='Time (years)',
                     row=2, col=1,
                     range=[-0.5, 7],
                     tickvals=np.arange(0, 6),
                     # ticktext=y_text,
                     showgrid=False)

    fig.update_xaxes(
        row=1, col=1,
        tickvals=np.arange(0, 6),
        showgrid=False)

    fig.update_yaxes(title_text='Fungicide curvature',
                     row=1, col=1,
                     # range=[-0.05, 1.05],
                     showgrid=False)

    fig.update_yaxes(title_text='Host control factor',
                     row=2, col=1,
                     range=[0.49, 1.01],
                     showgrid=False)

    return fig


def within_season(solution, no_control):
    """Within Season

    DPCs gradually getting worse over time

    Parameters
    ----------
    solution : dict
        output of simulations_run, for specific tactic e.g.
        single_data['spray_Y2_host_N']

    no_control : dict
        output of simulations_run, sprays 0 host off

    Returns
    -------
    _type_
        _description_

    Example
    -------
    >>>cs = Config('single', sprays=[0, 2], host_on = [False], n_years=20)
    >>>ws = simulations_run(cs)
    >>>within_season(ws['spray_Y2_host_N'], ws['spray_N_host_N'])
    """

    total_infection = solution['total_I']
    time = solution['t']

    ###
    traces = []
    shapez = []
    annotz = []
    ###

    N = total_infection.shape[1]

    colorscale = [
        x.hex for x in list(Color('green').range_to(Color('red'), N))
    ]

    selectors = [True if (i == 0 or i % 10 == 9)
                 else False for i in range(N+1)]

    for i in itertools.compress(range(N), selectors):

        col_use = colorscale[i]

        y_use = total_infection[:, i]

        line = dict(
            x=time,
            y=y_use,
            mode='lines',
            name=f'Year {i+1}',
            line={'color': col_use},
            showlegend=True,
        )

        traces.append(line)

    #
    #
    # no control

    y_NC = no_control['total_I'][:, 0]

    line = dict(
        x=time,
        y=y_NC,
        mode='lines',
        name='No Control',
        line={'color': 'black'},  # m , 'dash': 'dash'},
        opacity=0.5,
        showlegend=True,
    )

    traces.append(line)

    traces.reverse()

    fig = go.Figure(data=traces, layout=standard_layout(True))

    #
    #
    # ANNOTATIONS

    for text, pos in zip(
        ['Start of season', 'End of season'],
        [-0.1, 1.1]
    ):
        annotz.append(dict(
            x=pos,
            y=-0.25,
            text=text,
            showarrow=False,
            font=dict(
                color=GREY_LABEL
            ),
            xref='paper',
            yref='paper',
        ))

    fig.update_layout(
        shapes=shapez,
        annotations=annotz,
        legend=dict(
            x=0.6,
            y=1,
            traceorder="grouped",
            font=dict(size=12),
            bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(title='Time (degree-days)', showgrid=False),
        yaxis=dict(title='Disease severity (linear scale)', showgrid=False),
    )

    #
    #
    # INSET

    for line in traces:
        line_new = line
        line_new['y'] = [log10(ii/(1-ii)) for ii in line['y']]
        line_new['xaxis'] = 'x2'
        line_new['yaxis'] = 'y2'
        line_new['showlegend'] = False

        fig.add_trace(line_new)

        y_text = []
        for ii in [-3, -2, -1]:
            y_text.append(10**(ii)*1)
            y_text.append(10**(ii)*2)
            y_text.append(10**(ii)*5)

        y_vals = [log10(y/(1-y)) for y in y_text]

    #
    #
    # LAYOUT - INSET AXIS

    fig.update_layout(
        xaxis2=dict(
            domain=[0.2, 0.55],
            range=[1680, 1800],
            anchor='y2',
            tickfont=dict(size=14),
            showgrid=False,
        ),

        yaxis2=dict(
            title=dict(
                text='D.S. (logit scale)',
                font=dict(size=16),
            ),
            tickfont=dict(size=12),
            range=[log10(yy/(1-yy)) for yy in [0.0075, 0.02]],
            tickvals=y_vals,
            ticktext=y_text,
            domain=[0.4, 1],
            anchor='x2',
            showgrid=False,
        )
    )

    return fig


def compare_strategies_overview(multi, mutual, conf_multi, key1, key2):
    bds = [2.5, 97.5]

    col1 = 'rgba(50,50,50,1)'
    col2 = 'rgba(255,0,0,1)'
    col3 = 'rgba(0,255,0,1)'

    name1 = name_from_keys(key1, key2, '1')
    name2 = name_from_keys(key1, key2, '2')
    name3 = f"Difference: {name2.lower()} - {name1.split(' ')[1]}"

    fig = make_subplots(
        rows=4,
        cols=3,
        shared_xaxes=True,
        horizontal_spacing=0.15,
        column_widths=[0.25, 0.25, 0.5],
    )

    # FUNG DIST R1 C1
    dists_f1 = generate_dists(multi[key1][0], conf_multi, 10)
    trcs_fd1 = get_dist_traces(dists_f1, 'fung', col1, 5)

    dists_f2 = generate_dists(multi[key2][0], conf_multi, 10)
    trcs_fd2 = get_dist_traces(dists_f2, 'fung', col2, 5)

    fig.add_traces(trcs_fd1 + trcs_fd2, rows=1, cols=1)

    # FUNG DIST MEAN R2 C1
    fdm1 = dist_means_as_df(multi[key1], 'fung', conf_multi)
    trc_fdm1 = traces_with_uncertainty(
        fdm1, bds=bds, color=col1,
        name=name1,
        showlegend=True)

    fdm2 = dist_means_as_df(multi[key2], 'fung', conf_multi)
    trc_fdm2 = traces_with_uncertainty(
        fdm2, bds=bds, color=col2,
        name=name2,
        showlegend=True)

    fig.add_traces(trc_fdm1 + trc_fdm2, rows=2, cols=1)

    # FUNG DIFF DIST MEANS R3 C1
    fd_diff = fdm2 - fdm1
    trc_fd_diff = traces_with_uncertainty(fd_diff, bds=bds, color=col3,
                                          name=name3,
                                          showlegend=True,
                                          )

    fig.add_traces(trc_fd_diff, rows=3, cols=1)

    # HOST DIST R1 C2
    dists_h1 = generate_dists(multi[key1][0], conf_multi, 10)
    trcs_hd1 = get_dist_traces(dists_h1, 'host', col1, 5)

    dists_h2 = generate_dists(multi[key2][0], conf_multi, 10)
    trcs_hd2 = get_dist_traces(dists_h2, 'host', col1, 5)

    fig.add_traces(trcs_hd1 + trcs_hd2, rows=1, cols=2)

    # HOST DIST MEAN R2 C2
    hdm1 = dist_means_as_df(multi[key1], 'host', conf_multi)
    trc_hdm1 = traces_with_uncertainty(hdm1, bds=bds, color=col1)

    hdm2 = dist_means_as_df(multi[key2], 'host', conf_multi)
    trc_hdm2 = traces_with_uncertainty(hdm2, bds=bds, color=col2)

    fig.add_traces(trc_hdm1 + trc_hdm2, rows=2, cols=2)

    # HOST DIFF DIST MEANS R3 C2
    hd_diff = hdm2 - hdm1
    trc_hd_diff = traces_with_uncertainty(hd_diff, bds=bds, color=col3)

    fig.add_traces(trc_hd_diff, rows=3, cols=2)

    # Protective effect R4 C1,2
    without_host, without_fung = mutual_difference_traces(mutual,
                                                          bds,
                                                          col3)

    fig.add_traces(without_host, rows=4, cols=1)

    fig.add_traces(without_fung, rows=4, cols=2)

    # DS R1 C3
    df_ds1 = arrange_as_data_frame(multi[key1], 'dis_sev')
    trcs_ds1 = traces_with_uncertainty(df_ds1, bds=bds, color=col1)

    df_ds2 = arrange_as_data_frame(multi[key2], 'dis_sev')
    trcs_ds2 = traces_with_uncertainty(df_ds2, bds=bds, color=col2)

    fig.add_traces(trcs_ds1 + trcs_ds2, rows=1, cols=3)

    # YIELD R2 C3
    df_y1 = arrange_as_data_frame(multi[key1], 'yield_vec')
    trcs_y1 = traces_with_uncertainty(df_y1, bds=bds, color=col1)

    df_y2 = arrange_as_data_frame(multi[key2], 'yield_vec')
    trcs_y2 = traces_with_uncertainty(df_y2, bds=bds, color=col2)

    fig.add_traces(trcs_y1 + trcs_y2, rows=2, cols=3)

    # ECON DIFF R3&4 C3
    y1 = arrange_as_data_frame(multi[key1], 'econ')
    y2 = arrange_as_data_frame(multi[key2], 'econ')

    ydiff = y2 - y1

    trcs_diff = traces_with_uncertainty(ydiff, bds=bds, color=col3)

    fig.add_traces(trcs_diff, rows=4, cols=3)

    #
    #
    #
    # LAYOUT
    fig.update_layout(standard_layout(True))
    fig.update_layout(font=dict(size=6))
    fig.update_layout(legend=dict(
        x=0.43,
        y=-0.15,
        orientation='h',
        traceorder='normal',
        yanchor="top",
        xanchor="center",
    ))

    arrow_t = 0.95
    arrow_b = 0.70

    arrow_l = 0.2
    arrow_r = 0.51

    gap = 0.12

    annotz = [
        get_arrow_annotation(arrow_l, arrow_b, 0, 35, size=8),
        get_arrow_annotation(arrow_r, arrow_b, 0, 35, size=8),

        get_arrow_annotation(arrow_l, arrow_t, 0, 38, size=8),
        get_arrow_annotation(arrow_r, arrow_t, 0, 38, size=8),

        get_text_annotation(arrow_l, arrow_t + gap,
                            'Increasing<br>resistance', size=8),

        get_text_annotation(arrow_r, arrow_t + gap,
                            'Incr.<br>res.', size=8)
    ]

    fig.update_layout(annotations=annotz)

    #
    #
    # AXES
    fig.update_xaxes(title_text="Year", row=4, col=1, range=[0, 29])
    fig.update_xaxes(title_text="Year", row=4, col=2, range=[0, 25])
    fig.update_xaxes(title_text="Year", row=4, col=3, range=[0, 21])

    spacer_4 = 0.03
    spacer_3 = 0.02

    bottom = 0.02
    top = 1-bottom

    yc3 = [bottom+(top-bottom)*i/4 for i in range(1, 4)]
    yc4 = [bottom+(top-bottom)*i/3 for i in range(1, 3)]

    # col 1
    fig.update_yaxes(domain=[yc3[2]+spacer_4, top],
                     title_text='Distributions',
                     range=[0, 18],
                     row=1, col=1)

    fig.update_yaxes(domain=[yc3[1]+spacer_4, yc3[2]-spacer_4],
                     title_text='Distribution<br>means',
                     range=[0, 18],
                     row=2, col=1)

    fig.update_yaxes(domain=[yc3[0]+spacer_4, yc3[1]-spacer_4],
                     title_text='Difference in<br>dist. means',
                     row=3, col=1)

    fig.update_yaxes(domain=[bottom, yc3[0]-spacer_4],
                     title_text='Protective effect<br>(yield difference)',
                     row=4, col=1)

    # col 2
    fig.update_yaxes(domain=[yc3[2]+spacer_4, top],
                     range=[0.6, 1],
                     row=1, col=2)

    fig.update_yaxes(domain=[yc3[1]+spacer_4, yc3[2]-spacer_4],
                     range=[0.6, 1.01],
                     row=2, col=2)

    fig.update_yaxes(domain=[yc3[0]+spacer_4, yc3[1]-spacer_4],
                     row=3, col=2)

    fig.update_yaxes(domain=[bottom, yc3[0]-spacer_4],
                     row=4, col=2)

    # col 3
    fig.update_yaxes(domain=[yc4[1]+spacer_3, top],
                     title_text="Disease severity",
                     rangemode="tozero",
                     row=1, col=3)

    fig.update_yaxes(domain=[yc4[0]+spacer_3, yc4[1]-spacer_3],
                     title_text="Yield",
                     row=2, col=3)

    fig.update_yaxes(
        domain=[bottom, yc4[0]-spacer_3],
        title_text="Profit difference (£/ha)",
        # tickprefix='£',
        # showtickprefix="all",
        row=4, col=3
    )

    return fig


def compare_strategies_overview_2(multi, conf_multi, key1, key2):
    bds = [2.5, 97.5]

    col1 = 'rgba(50,50,50,1)'
    col2 = 'rgba(255,0,0,1)'
    col3 = 'rgba(0,255,0,1)'

    name1 = name_from_keys(key1, key2, '1')
    name2 = name_from_keys(key1, key2, '2')
    name3 = f"Difference: {name2.lower()} - {name1.split(' ')[1]}"

    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        horizontal_spacing=0.15,
        column_widths=[0.25, 0.25, 0.5],
    )

    # FUNG DIST R1 C1
    dists_f1 = generate_dists(multi[key1][0], conf_multi, 10)
    trcs_fd1 = get_dist_traces(dists_f1, 'fung', col1, 5)

    dists_f2 = generate_dists(multi[key2][0], conf_multi, 10)
    trcs_fd2 = get_dist_traces(dists_f2, 'fung', col2, 5)

    fig.add_traces(trcs_fd1 + trcs_fd2, rows=1, cols=1)

    # FUNG DIST MEAN R2 C1
    fdm1 = dist_means_as_df(multi[key1], 'fung', conf_multi)
    fdm2 = dist_means_as_df(multi[key2], 'fung', conf_multi)

    # FUNG DIFF DIST MEANS R3 C1
    fd_diff = fdm2 - fdm1
    trc_fd_diff = traces_with_uncertainty(fd_diff, bds=bds, color=col3,
                                          #   name=name3,
                                          #   showlegend=True,
                                          )

    fig.add_traces(trc_fd_diff, rows=3, cols=1)

    # HOST DIST R1 C2
    dists_h1 = generate_dists(multi[key1][0], conf_multi, 10)
    trcs_hd1 = get_dist_traces(dists_h1, 'host', col1, 5)

    dists_h2 = generate_dists(multi[key2][0], conf_multi, 10)
    trcs_hd2 = get_dist_traces(dists_h2, 'host', col1, 5)

    fig.add_traces(trcs_hd1 + trcs_hd2, rows=1, cols=2)

    # HOST DIFF DIST MEANS R3 C2
    hdm1 = dist_means_as_df(multi[key1], 'host', conf_multi)
    hdm2 = dist_means_as_df(multi[key2], 'host', conf_multi)

    hd_diff = hdm2 - hdm1
    trc_hd_diff = traces_with_uncertainty(hd_diff, bds=bds, color=col3)

    fig.add_traces(trc_hd_diff, rows=3, cols=2)

    # DS R1 C3
    df_ds1 = arrange_as_data_frame(multi[key1], 'dis_sev')
    trcs_ds1 = traces_with_uncertainty(
        df_ds1, bds=bds, color=col1,
        name=name1,
        showlegend=True)

    df_ds2 = arrange_as_data_frame(multi[key2], 'dis_sev')
    trcs_ds2 = traces_with_uncertainty(df_ds2, bds=bds, color=col2,
                                       name=name2,
                                       showlegend=True)

    fig.add_traces(trcs_ds1 + trcs_ds2, rows=1, cols=3)

    # YIELD R2 C3
    df_y1 = arrange_as_data_frame(multi[key1], 'yield_vec')
    trcs_y1 = traces_with_uncertainty(df_y1, bds=bds, color=col1)

    df_y2 = arrange_as_data_frame(multi[key2], 'yield_vec')
    trcs_y2 = traces_with_uncertainty(df_y2, bds=bds, color=col2)

    fig.add_traces(trcs_y1 + trcs_y2, rows=2, cols=3)

    # ECON DIFF R3 C3
    y1 = arrange_as_data_frame(multi[key1], 'econ')
    y2 = arrange_as_data_frame(multi[key2], 'econ')

    ydiff = y2 - y1

    trcs_diff = traces_with_uncertainty(ydiff, bds=bds, color=col3,
                                        name=name3,
                                        showlegend=True)

    fig.add_traces(trcs_diff, rows=3, cols=3)

    #
    #
    #
    # LAYOUT
    fig.update_layout(standard_layout(True))
    fig.update_layout(font=dict(size=6))
    fig.update_layout(legend=dict(
        x=0.43,
        y=-0.15,
        orientation='h',
        traceorder='normal',
        yanchor="top",
        xanchor="center",
    ))

    arrow_t = 0.95
    text_b = 0.53

    arrow_l = 0.18
    arrow_r = 0.51

    gap = 0.12

    annotz = [
        get_arrow_annotation(arrow_l, text_b + 0.02, 0, -100, size=8),

        get_arrow_annotation(arrow_r, arrow_t, 0, 115, size=8),

        get_text_annotation(arrow_l, text_b,
                            'Increasing<br>resistance', size=8),

        get_text_annotation(arrow_r, arrow_t + gap,
                            'Incr.<br>res.', size=8)
    ]

    fig.update_layout(annotations=annotz)

    #
    #
    # AXES
    fig.update_xaxes(title_text="Year", row=3, col=1, range=[0, 29])
    fig.update_xaxes(title_text="Year", row=3, col=2, range=[0, 25])
    fig.update_xaxes(title_text="Year", row=3, col=3, range=[0, 21])

    dom_gap = 0.03

    # col 1
    fig.update_yaxes(
        domain=[1/3 + dom_gap, 1],
        title_text='Distributions',
        range=[0, 20],
        row=1, col=1)

    fig.update_yaxes(
        title_text='Difference in<br>dist. means',
        row=3, col=1)

    # col 2
    fig.update_yaxes(
        domain=[1/3 + dom_gap, 1],
        range=[0.6, 1],
        row=1, col=2)

    fig.update_yaxes(
        row=3, col=2)

    # col 3
    fig.update_yaxes(
        title_text="Disease severity",
        rangemode="tozero",
        row=1, col=3)

    fig.update_yaxes(
        title_text="Yield",
        row=2, col=3)

    fig.update_yaxes(
        title_text="Profit difference (£/ha)",
        row=3, col=3
    )

    return fig
