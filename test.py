import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from main import make_plot
import numpy as np
# Setup app
app = dash.Dash()
n_points = 140
alpha = 0.6
beta = 0.1
x_from = 0
x_to = 5 


# Make app layout
app.layout = html.Div(
    [
        html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Number of points:'),
                    dcc.Slider(
                        id='n_points',
                        min=0,
                        max=200,
                        value=50,
                    ),
                ]),
                html.Div([
                    html.Label('Alpha'),
                    dcc.Slider(
                        id='alpha',
                        min=-50,
                        max=50,
                        value=2,
                    ),
                ]),
                html.Div([
                    html.Label('Beta'),
                    dcc.Slider(
                        id='beta',
                        min=-50,
                        max=50,
                        value=4,
                    )
                ]),
            ],
                className='four columns'
            ),
        ],
            className='row',
            style={'margin-bottom': '10'}
        ),
        html.Div([
            html.Div([
                html.Label('settings:'),
                html.Div([
                    dcc.RadioItems(
                        id='iv_selector',
                        options=[
                            {'label': 'Knapp1 ', 'value': True},
                            {'label': 'Knapp2 ', 'value': False},
                        ],
                        value=True,
                        labelStyle={'display': 'inline-block'},
                    ),
                    dcc.RadioItems(
                        id='calendar_selector',
                        options=[
                            {'label': 'Knapp3 ', 'value': True},
                            {'label': 'Knapp4 ', 'value': False},
                        ],
                        value=True,
                        labelStyle={'display': 'inline-block'},
                    )
                ],
                    style={'display': 'inline-block', 'margin-right': '10', 'margin-bottom': '10'}
                )
            ],
                className='six columns',
                style={'display': 'inline-block'}
            ),
        ],
            className='row'
        ),
        html.Div([
            dcc.Graph(id='iv_surface', style={'max-height': '600', 'height': '60vh'}),
        ],
            className='row',
            style={'margin-bottom': '20'}
        )
    ],
    style={
        'width': '85%',
        'max-width': '1200',
        'margin-left': 'auto',
        'margin-right': 'auto',
        'font-family': 'overpass',
        'background-color': '#F3F3F3',
        'padding': '40',
        'padding-top': '20',
        'padding-bottom': '20',
    },
)




# Make main surface plot
@app.callback(Output('iv_surface', 'figure'),
              [Input('n_points', 'value'),
              Input('alpha', 'value'),
              Input('beta', 'value')],
              [State('iv_surface', 'relayoutData')])
def make_surface_plot(n_points,alpha,beta,iv_surface_layout):

    X,Z,first = make_plot(x_from,x_to,n_points,alpha,beta)
    trace1 = {
        "type": "surface",
        'x': X,
        'y': Z,
        'z': first,
        #'intensity': first,
        'colorscale': 'Viridis',
        #"lighting": {
        #    "ambient": 1,
        #    "diffuse": 0.9,
        #    "fresnel": 0.5,
        #    "roughness": 0.9,
        #    "specular": 2
        }#,
       # "reversescale": True,
    

    layout = {
        "title": "Title",
        'margin': {
            'l': 10,
            'r': 10,
            'b': 10,
            't': 60,
        },
        'paper_bgcolor': '#FAFAFA',
        "hovermode": "closest",
        "scene": {
            "aspectmode": "manual",
            "aspectratio": {
                "x": 2,
                "y": 2,
                "z": 1
            },
            'camera': {
                'up': {'x': 0, 'y': 0, 'z': 1},
                'center': {'x': 0, 'y': 0, 'z': 0},
                'eye': {'x': 1, 'y': 2, 'z': 2},
            },
            "xaxis": {
                "showbackground": False
                #"backgroundcolor": "rgb(230, 230,230)",
                #"gridcolor": "rgb(255, 255, 255)",
                #"zerolinecolor": "rgb(255, 255, 255)"
            },
            "yaxis": {
                "showbackground": False
                #"backgroundcolor": "rgb(230, 230,230)",
                #"gridcolor": "rgb(255, 255, 255)",
                #"zerolinecolor": "rgb(255, 255, 255)"
            },
            "zaxis": {
                "showbackground": False
                #"backgroundcolor": "rgb(230, 230,230)",
                #"gridcolor": "rgb(255, 255, 255)",
                #"zerolinecolor": "rgb(255, 255, 255)"
            }
        },
    }
    if (iv_surface_layout is not None):

        up = iv_surface_layout['scene.camera']['up']
        center = iv_surface_layout['scene.camera']['center']
        eye = iv_surface_layout['scene.camera']['eye']
        layout['scene']['camera']['up'] = up
        layout['scene']['camera']['center'] = center
        layout['scene']['camera']['eye'] = eye


    data = [trace1]
    figure = dict(data=data, layout=layout)
    return figure



# Main
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True,port=8031)