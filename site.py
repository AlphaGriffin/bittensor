import os, sys, time, datetime, collections
from decimal import Decimal as D
import numpy as np
from timeit import default_timer as timer

import pandas as pd

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import colorlover as cl  # fucking really????

import ag.bittensor.utils.options as options
from ag.bittensor.ai.make_data import MakeData
import ag.bittensor.utils.talib as talib

import flask

# ////////////// Build Globals
config = options.Options('config/access_codes.yaml')
datasmith = MakeData(config)
TA = talib.TALib(config)

# PRE-SETUP DATA INIT
# make dropdown data:
dropdownelements = []
for i in datasmith.all_pairs:
    if 'USDT' in i:
        coin = i[:-9]
    else:
        coin = i[:-8]
    pair = i[:-4]
    item = {
        'label': '{} - {}'.format(coin, pair),
        'value': i
    }
    dropdownelements.append(item)

# get all available TA conditions
taelements = []
for i, e in enumerate(TA.available):
    taelements.append({
        'label': e,
        'value': e
    })

# ////////////// START SETUP
colorscale = cl.scales['9']['qual']['Paired']
colors = {
    'background': '#111316',
    'text': '#7FDBFF'
}

banner = html.H1(
    children = 'AlphaGriffin Technical Analysis',
    style = {
        'textAlign': 'center',
        'color': colors['text']
    }
)

banner2 = html.Div(
    children='''{}'''.format(datetime.datetime.now()),
    style = {
        'textAlign': 'center',
        'color': colors['text']
    }
)

# ////////////// MAKE BUTTONS
dropbox = dcc.Dropdown(
    id='top-dropdown',
    options=[
        x for x in dropdownelements
        ],
    value='BTC_USDT.csv'
)

checkboxes = dcc.Checklist(
    id='ta-checks',
    options=[
        x for x in taelements
    ],
    values=['Moving Average'],
    style={
        'color': '#ffffff'
    }
)

buttons = html.Div(
    children = [
        html.Label('Available Pairs', style={'color': '#ffffff'}),
        dropbox,
        html.Label('Available TA Indicators', style={'color': '#ffffff'}),
        checkboxes
    ],
    style={'backgroundColor': '#333333'}
)
# ////////////// /Buttons

# ////////////// MAKE GRAPH STARTING GRAPH
FIG = {
    'data': [],
    'layout': {
        'title': 'LoadingScreen'
    }
}

randomfile = datasmith.random_filename
datasmith.dataframe = randomfile
datasmith.candles = '15T'
start_df = datasmith.candles
MASTERDF = start_df
FIG['data'].append({
    'x': start_df.index,
    'y': start_df['Volume'],
    'type': 'bar',
    'name': 'Volume'
})

graph = dcc.Graph(
    id='main-graph',
    figure = FIG
)

graphdiv = html.Div(id='graphs', children=[graph])
subgraphdiv = html.Div(id='subgraphs', children=[graph])


# ////////////// /graph
mainwindow = html.Div([buttons, graphdiv, subgraphdiv])

# ////////////// START APP
app = dash.Dash()
app.css.append_css({'external_url': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'})
app.layout = html.Div(
    style={
        "backgroundColor": colors['background'],
    },
    children=[
        banner,
        banner2,
        mainwindow,
    ]
)

@app.callback(Output('subgraphs', 'children'), [Input('top-dropdown', 'value'), Input('ta-checks', 'values')])
def subupdate(pair, tachecks):
    graphs = []
    df = MASTERDF
    for x in tachecks:
        if 'Realitive Strength Index' in x:
            indicator = TA.available[x](df)
            # start a new graph
            data = {
                'x': df.index,
                'y': indicator,
                'type': 'scatter',
                'name': x
            }
            topline = {
                'x': df.index,
                'y': np.repeat(70, len(df)),
                'type': 'line',
                'name': 'overbought'
            }
            botline = {
                'x': df.index,
                'y': np.repeat(30, len(df)),
                'type': 'line',
                'name': 'oversold'
            }

            rsi = dcc.Graph(id=x, figure={
                'data': [data, topline, botline],
                'layout': {
                    # 'title': '{}'.format(x),
                    'margin': {
                        'l': 100,
                        'r': 100,
                        't': 20,
                        'b': 10
                        },
                    'legend': {'x': 0}
                }
            })
            graphs.append(rsi)

        if 'Commodity Channel Index' in x:
            indicator = TA.available[x](df)
            # start a new graph
            data = {
                'x': df.index,
                'y': indicator,
                'type': 'scatter',
                'name': x
            }
            topline = {
                'x': df.index,
                'y': np.repeat(100, len(df)),
                'type': 'line',
                'name': 'overbought'
            }
            botline = {
                'x': df.index,
                'y': np.repeat(-100, len(df)),
                'type': 'line',
                'name': 'oversold'
            }

            rsi = dcc.Graph(id=x, figure={
                'data': [data, topline, botline],
                'layout': {
                    # 'title': '{}'.format(x),
                    'margin': {
                        'l': 100,
                        'r': 100,
                        't': 10,
                        'b': 10
                        },
                    'legend': {'x': 0}
                }
            })
            graphs.append(rsi)

        """
        z = dcc.Graph(
            id='stuff'+str(x),
            figure={
                'data': [{'x': list(range(5)), 'y':[2,4,1,5,3], 'type': 'line'}],
                'layout': {
                    'margin': {'b': 15, 'r': 100, 'l': 60, 't': 10},
                    'legend': {'x': 0}
                }
            }
        )
        graphs.append(z)
        """
    return graphs

@app.callback(Output('graphs', 'children'), [Input('top-dropdown', 'value'), Input('ta-checks', 'values')])
def mainupdate(pair, tachecks):
    timeframe = '1H'
    datasmith.dataframe = pair
    datasmith.candles = timeframe
    MASTERDF = df = datasmith.candles

    # setup main graph
    candlestick = {
        'x': df.index,
        'open': df['Open'],
        'high': df['High'],
        'low': df['Low'],
        'close': df['Close'],
        'volume': df['Volume'],
        'type': 'candlestick',
        'name': pair[:-4],
        'legendgroup': pair[:-4],
        'increasing': {'line': {'color': colorscale[0]}},
        'decreasing': {'line': {'color': colorscale[1]}}
    }
    FIG = {
        'data': [],
        'layout': {
            'title': '{}'.format(pair[:-4]),
            'margin': {
                'l': 100,
                'r': 100,
                't': 40,
                'b': 10
                },
            'legend': {'x': 0}
        }
    }
    FIG['data'].append(candlestick)
    for i in tachecks:
        if 'Exp Moving Average' in i or 'Moving Average' in i or 'Average True Range' in i:
            indicator = TA.available[i](df)
            data = {
                'x': df.index,
                'y': indicator,
                'type': 'line',
                'name': i
            }
            FIG['data'].append(data)
        if 'Bollinger Bands' in i:
            indicator = TA.available[i](df)
            for index, j in enumerate(indicator):
                # print(j.values)
                data = {
                    'x': df.index,
                    'y': j.values,
                    'type': 'scatter',
                    'name': '{} {}'.format(i, index)
                }
                FIG['data'].append(data)

    return dcc.Graph(id=pair[:-4], figure=FIG)


if __name__ == '__main__':
    app.run_server(debug=True)
