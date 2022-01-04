"""
License
-------
Copyright (C) 2021  - David Fernández Castellanos

You can use this software, redistribute it, and/or modify it under the 
terms of the Creative Commons Attribution 4.0 International Public License.

"""

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import State, Input, Output
import plotly.io as pio
import numpy as np
import datetime
from datetime import datetime as dt

from model import run_model
from plot import get_country_data, plot_model_results, plot_country_data

# pylint: disable=E0102

title = "Statistical model of the COVID-19 vaccination campaign"

pio.templates.default = "plotly_white"

avail_countries, country_data = get_country_data()

storage_type = "memory"

tab_style = {"padding-top": 7}

tab_selected_style = {"padding-top": 7}

tabs_styles = {"height": "44px"}


def generate_population_controls():

    return html.Div(
        id="population-controls",
        children=[
            html.Div(id="output-p-pro-value"),
            dcc.RangeSlider(
                id="slider-p-pro",
                min=0.0,
                max=100,
                value=[40, 50],
                marks={"0": "0%", "100": "100%"},
                step=1,
                pushable=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-p-pro", storage_type=storage_type),
            html.Br(),
            html.Div(id="output-p-anti-value"),
            dcc.RangeSlider(
                id="slider-p-anti",
                min=0.0,
                max=100,
                value=[15, 25],
                marks={"0": "0%", "100": "100%"},
                step=1,
                pushable=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-p-anti", storage_type=storage_type),
            html.Br(),
            html.Div(id="agnostics-pct-msg"),
            html.Br(),
            html.Div(id="output-p-soft-no-value"),
            html.Div(id="output-pressure-value"),
            dcc.RangeSlider(
                id="slider-pressure",
                min=0.0,
                max=0.2,
                value=[0.02, 0.05],
                marks={"0": "0", "0.2": "0.2"},
                step=0.005,
                pushable=0.005,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-pressure", storage_type=storage_type),
        ],
    )


def generate_vaccine_controls():

    return html.Div(
        id="vaccine-controls",
        children=[
            html.Div(id="output-nv0-value"),
            dcc.RangeSlider(
                id="slider-nv0",
                min=0.02,
                max=3.0,
                value=[0.2, 0.3],
                marks={"0.02": "0.02%", "3": "3%"},
                step=0.02,
                pushable=0.02,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-nv0", storage_type=storage_type),
            html.Br(),
            html.Div(id="output-tau-value"),
            dcc.RangeSlider(
                id="slider-tau",
                min=1,
                max=20,
                value=[3, 5],
                marks={"1": "1 week", "20": "20 weeks"},
                step=0.5,
                pushable=0.5,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-tau", storage_type=storage_type),
            html.Br(),
            html.Div(id="output-nvmax-value"),
            dcc.RangeSlider(
                id="slider-nvmax",
                min=0.1,
                max=10,
                value=[4, 7],
                marks={"0.1": "0.1%", "10": "10%"},
                step=0.1,
                pushable=0.1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-nvmax", storage_type=storage_type),
        ],
    )


def generate_sampling_controls():

    return html.Div(
        id="sampling-controls",
        children=[
            html.Div(id="output-nrep-value"),
            dcc.Input(
                id="input-nrep",
                type="number",
                value=100,
                min=1,
                max=10000,
                step=1,
                debounce=False,
            ),
            dcc.Store(id="store-nrep", storage_type=storage_type),
            # html.Br(),
            # html.Br(),
            html.Div(id="output-N-value", style={"display": "none"}),
            dcc.Input(
                id="input-N",
                type="number",
                value=50000,
                min=1,
                max=1000000,
                step=10,
                debounce=False,
                style={"display": "none"},
            ),
            dcc.Store(id="store-N", storage_type=storage_type),
            html.Br(),
            html.Br(),
            html.Div(id="output-CI-value"),
            dcc.Slider(
                id="slider-CI",
                min=0.0,
                max=100.0,
                value=95,
                marks={"0": "0%", "100": "100%"},
                step=5,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-CI", storage_type=storage_type),
        ],
    )


def generate_country_and_date_controls():

    return html.Div(
        id="contry-date-controls",
        children=[
            html.P("Date range"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=dt(2020, 12, 30),
                end_date=dt.today() + datetime.timedelta(days=90),
                display_format="MMM Do, YY",
                initial_visible_month=dt.today(),
            ),
            dcc.Store(id="store-date-range", storage_type=storage_type),
            html.Br(),
            html.Br(),
            html.P("Add country"),
            dcc.Dropdown(
                id="country-select",
                options=[{"label": i, "value": i} for i in avail_countries],
                value=["Germany", "United States", "Russia"],
                multi=True,
            ),
            dcc.Store(id="store-countries", storage_type=storage_type),
        ],
    )


def generate_plots_section():

    return html.Div(
        id="plot_header",
        children=[
            dcc.Graph(
                id="plot_grid",
                config={
                    "toImageButtonOptions": {
                        "format": "svg",
                        "filename": "vaccination_model",
                        "scale": 2,
                    },
                    "displaylogo": False,
                    "showTips": True,
                },
                style={
                    "width": "127vh",
                    "height": "80vh",
                    # "display": "inline-block",
                    # "overflow": "hidden",
                    # "position": "absolute",
                    # "top": "50%",
                    # "left": "50%",
                    # "transform": "translate(-50%, -50%)"
                },
            ),
            dcc.Store(id="store-model-results", storage_type=storage_type),
        ],
    )


def generate_model_explanation():

    return html.Iframe(
        src="assets/model_explanation.html", style={"height": "100vh", "width": "100%"}
    )


def generate_about():

    s = """   
        This project has been created by David Fernández Castellanos.
                
        ## Related links:
        -   The author's website: [https://www.davidfcastellanos.com](https://www.davidfcastellanos.com)                    
        -   The source code of the web App and the model back-end: [https://github.com/kastellane/COVID19-Vaccination-Model](https://github.com/kastellane/COVID19-Vaccination-Model)
        -   An associated blog post with extra information: [https://www.davidfcastellanos.com/covid-19-vaccination-model](https://www.davidfcastellanos.com/covid-19-vaccination-model)
        -   Real world data: [https://github.com/owid/covid-19-data](https://github.com/owid/covid-19-data)
            
        """
    return html.Div(
        id="text-explanation",
        children=[
            html.Br(),
            html.Br(),
            dcc.Markdown(s),
        ],
    )


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = title
server = app.server

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H2(
                    title,
                    style={"color": "#2c8cff"},
                ),
            ],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[
                html.Div(
                    dcc.Tabs(
                        children=[
                            dcc.Tab(
                                label="Population",
                                children=[html.Br(), generate_population_controls()],
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Vaccines",
                                children=[html.Br(), generate_vaccine_controls()],
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Sampling",
                                children=[html.Br(), generate_sampling_controls()],
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                        ],
                        style=tabs_styles,
                    ),
                ),
                html.Center(
                    [
                        html.Button("Run", id="run-button"),
                        dcc.Loading(
                            id="ls-loading-2",
                            children=[html.Div([html.Div(id="ls-loading-output-2")])],
                            type="circle",
                            style={"scale": "1.5"},
                        ),
                    ]
                ),
                html.Hr(style={"height": "2px"}),
                generate_country_and_date_controls(),
                html.Br(),
                html.Div(id="error-msg", style={"color": "red"}),
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                dcc.Tabs(
                    children=[
                        dcc.Tab(
                            label="Results",
                            children=[
                                generate_plots_section(),
                            ],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                        dcc.Tab(
                            label="Model explanation",
                            children=[generate_model_explanation()],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                        dcc.Tab(
                            label="About",
                            children=[generate_about()],
                            style=tab_style,
                            selected_style=tab_selected_style,
                        ),
                    ],
                    style=tabs_styles,
                ),
            ],
        ),
    ],
)


@app.callback(
    # update graph and spinner
    Output("plot_grid", "figure"),
    Output("ls-loading-output-2", "children"),
    # population message: error and agnostic percentage
    Output("agnostics-pct-msg", "children"),
    Output("error-msg", "children"),
    # update the stored model results
    Output("store-model-results", "data"),
    # run button
    Input("run-button", "n_clicks"),
    # countries and dates
    Input("country-select", "value"),
    # population parameters
    State("store-p-pro", "data"),
    State("store-p-anti", "data"),
    State("store-pressure", "data"),
    # vaccinations parameters
    State("store-tau", "data"),
    State("store-nv0", "data"),
    State("store-nvmax", "data"),
    # samping
    State("store-CI", "data"),
    State("store-nrep", "data"),
    State("store-N", "data"),
    # date range
    State("store-date-range", "data"),
    # model result
    State("store-model-results", "data"),
)
def update_figures(
    # button input
    n_clicks,
    # countries
    selected_countries,
    # populatio parameters
    p_pro_bounds,
    p_anti_bounds,
    pressure_bounds,
    # vaccinations parameters
    tau_bounds,
    nv_0_bounds,
    nv_max_bounds,
    # samping
    CI,
    n_rep,
    N,
    date_range,
    # model results
    model_results,
):

    # we can re-use previous model results if we must update the figure but
    # only the country selection changed. For that, we can check the callback
    # contex and see if the trigger was the run button being pressed. If not,
    # we can read the stored model results and resuse them
    ctx = dash.callback_context.triggered
    assert len(ctx) == 1
    update_model = ctx[0]["prop_id"] == "run-button.n_clicks"

    # another options is using the stored last model run, we define a seed so that the
    # values obtained from calling sample_param_combinations are
    # repeated when no changes from the GUI components are mande.
    # This means that the parameters to run_model_sampling
    # repeat exactly so we can leverage this by using the @functools.lru_cache
    # decorator on run_model_sampling. In this way the results are automatically
    # cached and reused when the exactly the same input repeats
    np.random.seed(12345)

    # NOTE: although the second option shoulb be enough, we enable the first option,
    # since when the app is deployed in Heroku the second one does not caches the results.
    # This means that a repeated call to the model will be re-run, but if we modify
    # the selection of countries it won't run again

    # default output messages
    msg_agnostics_pct = "Agnosticts: "
    msg_error = ""

    # check n_clicks to make sure that the run button has already been pressed once
    if update_model and n_clicks is not None:

        max_running_time = 30

        model_results, msg_error, msg_agnostics_pct = run_model(
            # populatio parameters
            p_pro_bounds,
            p_anti_bounds,
            pressure_bounds,
            # vaccinations parameters
            tau_bounds,
            nv_0_bounds,
            nv_max_bounds,
            # samping
            CI,
            n_rep,
            N,
            date_range,
            max_running_time,
        )

    fig = plot_model_results(model_results, CI)

    initialized = model_results is None
    plot_country_data(fig, selected_countries, country_data, initialized)

    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))  # height=400, width=1100)

    return fig, None, msg_agnostics_pct, msg_error, model_results


@app.callback(
    Output("store-p-pro", "data"),
    Output(component_id="output-p-pro-value", component_property="children"),
    Input(component_id="slider-p-pro", component_property="value"),
    State("store-p-pro", "data"),
)
def update_output_div(values, data):
    data = values
    return data, f"Pro-vaccines: {values[0]} - {values[1]}%"


@app.callback(
    Output("store-p-anti", "data"),
    Output(component_id="output-p-anti-value", component_property="children"),
    Input(component_id="slider-p-anti", component_property="value"),
    State("store-p-anti", "data"),
)
def update_output_div(values, data):
    data = values
    return data, f"Anti-vaccines: {values[0]} - {values[1]}%"


@app.callback(
    Output("store-pressure", "data"),
    Output(component_id="output-pressure-value", component_property="children"),
    Input(component_id="slider-pressure", component_property="value"),
    State("store-pressure", "data"),
)
def update_output_div(values, data):
    data = values
    return data, f"Pressure on the agnostics: {values[0]} - {values[1]}"


@app.callback(
    Output("store-nv0", "data"),
    Output(component_id="output-nv0-value", component_property="children"),
    Input(component_id="slider-nv0", component_property="value"),
    State("store-nv0", "data"),
)
def update_output_div(values, data):
    data = values
    return data, f"Initial stock: {values[0]} - {values[1]}% of the pop."


@app.callback(
    Output("store-tau", "data"),
    Output(component_id="output-tau-value", component_property="children"),
    Input(component_id="slider-tau", component_property="value"),
    State("store-tau", "data"),
)
def update_output_div(values, data):
    data = values
    return data, f"Duplication time: {values[0]} - {values[1]} weeks"


@app.callback(
    Output("store-nvmax", "data"),
    Output(component_id="output-nvmax-value", component_property="children"),
    Input(component_id="slider-nvmax", component_property="value"),
    State("store-nvmax", "data"),
)
def update_output_div(values, data):
    data = values
    return data, f"Weekly arrival limit: {values[0]} - {values[1]}% of the pop."


@app.callback(
    Output("store-CI", "data"),
    Output(component_id="output-CI-value", component_property="children"),
    Input(component_id="slider-CI", component_property="value"),
    State("store-CI", "data"),
)
def update_output_div(value, data):
    data = value
    return data, f"Confidence interval: {value}%"


@app.callback(
    Output("store-nrep", "data"),
    Output(component_id="output-nrep-value", component_property="children"),
    Input(component_id="input-nrep", component_property="value"),
    State("store-nrep", "data"),
)
def update_output_div(value, data):
    data = value
    return data, f"Number of Monte Carlo runs: {value}"


@app.callback(
    Output("store-N", "data"),
    Output(component_id="output-N-value", component_property="children"),
    Input(component_id="input-N", component_property="value"),
    State("store-N", "data"),
)
def update_output_div(value, data):
    data = value
    return data, f"Population size: {value}"


@app.callback(
    Output("store-date-range", "data"),
    Input("date-picker-select", "start_date"),
    Input("date-picker-select", "end_date"),
    State("store-date-range", "data"),
)
def update_output_div(start_date, end_date, data):
    data = data or dict()
    data["start_date"] = start_date
    data["end_date"] = end_date
    return data


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(host='0.0.0.0', port=80)
