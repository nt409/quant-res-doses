MARKER_COLOURS = [
    'rgb(50,50,50)',  # 'black',
    'rgb(255,0,0)',  # 'red',
    'rgb(255,150,50)',  # 'orange',
    'rgb(0,0,255)',  # 'blue',
    'rgb(255,150,255)',  # 'pink',
    'rgb(0,255,0)',  # 'green',
    'rgb(130,0,255)',  # 'purple',
    'rgb(0,255,200)',  # '',
]*2


TYPE_PLOT_MAP = {'yield': 'yield_vec',
                 'DS': 'dis_sev',
                 'econ': 'econ'}


Y_LABEL_MAP = {'yield': 'Yield (tons/ha)',
               'DS': 'Disease severity',
               'econ': 'Economic yield (£/ha)'}

PERCENTILES = [25, 75]
# PERCENTILES  = [2.5,97.5]

HOST_COLOUR = 'rgba(144,238,144,1)'
FUNG_COLOUR = 'rgba(153,204,255,1)'


GREY_LABEL = "rgb(110,110,110)"

LIGHT_GREY_TEXT = "rgb(150,150,150)"

FILL_OPACITY = 0.2


KEY_TO_LONGNAME = {
    'spray_N_host_N': 'Sprays: 0, host: not res.',
    'spray_N_host_Y': 'Sprays: 0, host: res.',
    'spray_Y1_host_N': 'Sprays: 1, host: not res.',
    'spray_Y1_host_Y': 'Sprays: 1, host: res.',
    'spray_Y2_host_N': 'Sprays: 2, host: not res.',
    'spray_Y2_host_Y': 'Sprays: 2, host: res.',
    'spray_Y3_host_N': 'Sprays: 3, host: not res.',
    'spray_Y3_host_Y': 'Sprays: 3, host: res.'
}

KEY_TO_SHORTNAME = {
    'spray_N_host_N': 'Sprays: 0, host: NR',
    'spray_N_host_Y': 'Sprays: 0, host: R',
    'spray_Y1_host_N': 'Sprays: 1, host: NR',
    'spray_Y1_host_Y': 'Sprays: 1, host: R',
    'spray_Y2_host_N': 'Sprays: 2, host: NR',
    'spray_Y2_host_Y': 'Sprays: 2, host: R',
    'spray_Y3_host_N': 'Sprays: 3, host: NR',
    'spray_Y3_host_Y': 'Sprays: 3, host: R'
}


KEY_COLOURS = [
    'rgb(255,180,0)',  # 'red1',
    'rgb(255,120,0)',  # 'red2',
    'rgb(255,60,0)',  # 'red3',
    'rgb(255,0,0)',  # 'red4',
    'rgb(200,200,200)',  # 'black',
    'rgb(150,150,150)',  # 'black',
    'rgb(100,100,100)',  # 'black',
    'rgb(0,0,0)',  # 'black',
]


KEY_ATTRS = {
    'spray_N_host_N':  dict(name='Sprays: 0, host: NR', sprays='0', host='N', dash='dot',    colour=KEY_COLOURS[0], symbol='cross'),
    'spray_N_host_Y':  dict(name='Sprays: 0, host: R',  sprays='0', host='Y', dash='solid',  colour=KEY_COLOURS[4], symbol='cross'),
    'spray_Y1_host_N': dict(name='Sprays: 1, host: NR', sprays='1', host='N', dash='dot',    colour=KEY_COLOURS[1], symbol='diamond'),
    'spray_Y1_host_Y': dict(name='Sprays: 1, host: R',  sprays='1', host='Y', dash='solid',  colour=KEY_COLOURS[5], symbol='diamond'),
    'spray_Y2_host_N': dict(name='Sprays: 2, host: NR', sprays='2', host='N', dash='dot',    colour=KEY_COLOURS[2], symbol='x'),
    'spray_Y2_host_Y': dict(name='Sprays: 2, host: R',  sprays='2', host='Y', dash='solid',  colour=KEY_COLOURS[6], symbol='x'),
    'spray_Y3_host_N': dict(name='Sprays: 3, host: NR', sprays='3', host='N', dash='dot',    colour=KEY_COLOURS[3], symbol='circle'),
    'spray_Y3_host_Y': dict(name='Sprays: 3, host: R',  sprays='3', host='Y', dash='solid',  colour=KEY_COLOURS[7], symbol='circle')
}

TRAIT_FULLNAME = {
    'fung': 'Fungicide',
    'host': 'Host',
}


# default is half page
PLOT_WIDTH = 600
PLOT_HEIGHT = 400
