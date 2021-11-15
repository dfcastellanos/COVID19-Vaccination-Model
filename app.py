import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from datetime import datetime as dt


import model


# pylint: disable=E0102

pio.templates.default = "plotly_white"


def get_country_data():

    df = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"])

    columns_to_keep = [
        "location",
        "date",
        "people_fully_vaccinated_per_hundred",
        "daily_vaccinations_per_million",
    ]

    df = df.loc[:, columns_to_keep]

    avail_countries = df["location"].unique()
    country_data = pd.pivot_table(
        df,
        columns="location",
        values=[
            "people_fully_vaccinated_per_hundred",
            "daily_vaccinations_per_million",
        ],
        index="date",
    )

    return avail_countries, country_data


avail_countries, country_data = get_country_data()


def description_card():

    return html.Div(
        id="description-card",
        children=[
            html.H5("Monte Carlo model of vaccination campaings"),
            html.H3(
                "Welcome to the Dashboard of the Monte Carlo model of vaccination campaings"
            ),
            html.Div(
                id="intro",
                children="Add here a brief description about these controls.",
            ),
        ],
    )


def generate_control_card():

    return html.Div(
        id="control-card",
        children=[
            html.P("Population views on vaccines"),
            html.Div(id="output-p-yes-value"),
            dcc.Slider(
                id="slider-p-yes",
                min=0.0,
                max=100,
                value=70,
                marks={"0": "0%", "100": "100%"},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id="output-p-hard-no-value"),
            dcc.Slider(
                id="slider-p-hard-no",
                min=0.0,
                max=100,
                value=10,
                marks={"0": "0%", "100": "100%"},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id="output-p-soft-no-value"),
            html.Br(),
            html.Div(id="output-pressure-value"),
            dcc.Slider(
                id="slider-pressure",
                min=0.0,
                max=10,
                value=5,
                marks={"0": "0%", "10": "10%"},
                step=0.5,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Br(),
            html.P("Date range"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=dt(2020, 12, 30),
                end_date=dt.today(),
                display_format="MMM Do, YY",
                initial_visible_month=dt.today(),
            ),
            html.Br(),
            html.Br(),
            html.P("Add country"),
            dcc.Dropdown(
                id="country-select",
                options=[{"label": i, "value": i} for i in avail_countries],
                value=[],
                multi=True,
            ),
        ],
    )


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
    ],
    title="Statistical vaccination model",
)


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # plots section
                html.Div(
                    id="plot_header",
                    children=[
                        html.B("Model results"),
                        html.Hr(),
                        dcc.Graph(
                            id="plot_grid",
                            style={
                                "width": "100vh",
                                "height": "100vh",
                                # "display": "inline-block",
                                # "overflow": "hidden",
                                "position": "absolute",
                                # "top": "50%",
                                # "left": "50%",
                                # "transform": "translate(-50%, -50%)"
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def add_line(fig, s, color, name, row, col):

    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s,
            mode="lines",
            legendgroup=name,
            name=name,
            line_shape="spline",
            showlegend=False,
            line=dict(color=color),
        ),
        row=row,
        col=col,
    )

    fig.add_annotation(
        xref="paper",
        x=s.index[-1],
        y=s[-1],
        xanchor="left",
        yanchor="middle",
        text=name,
        font=dict(family="Arial", size=14, color=color),
        showarrow=False,
        row=row,
        col=col,
    )

    return fig


@app.callback(
    Output("plot_grid", "figure"),
    [
        Input("slider-p-yes", "value"),
        Input("slider-p-hard-no", "value"),
        Input("slider-pressure", "value"),
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("country-select", "value"),
    ],
)
def update_figures(
    p_yes, p_hard_no, pressure, start_date, end_date, selected_countries
):

    start_date = dt.strptime(start_date.split("T")[0], "%Y-%m-%d")
    end_date = dt.strptime(end_date.split("T")[0], "%Y-%m-%d")

    N = 5000

    # sliders use values 0-100
    p_yes /= 100
    p_hard_no /= 100
    pressure /= 100

    max_delivery = 0.05
    a = 0.0007
    F = lambda t: min(a * np.exp(0.2 * t / 7), max_delivery) * N

    nv_purchased = 1.5 * N

    data = model.run(
        p_yes, p_hard_no, pressure, nv_purchased, start_date, end_date, F, N
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "People vaccinated per hundred",
            "Daily vaccinations per million",
            "Total dosis received / purchsed",
            "Dosis in stock per hundred",
        ),
    )

    g = data["people_fully_vaccinated_per_hundred"]
    fig = add_line(fig, g, "royalblue", "Model", 1, 1)

    # multiply by 2 to take into account that in the real world data, for most
    # of the countries (at least for EU) a full vaccination counts as
    # 2 units
    g = data["daily_vaccinations_per_million"].resample("6D").agg(np.mean) * 2
    fig = add_line(fig, g, "royalblue", "Model", 1, 2)

    g = data["ratio_vaccines_received_over_purchased"]
    fig = add_line(fig, g, "royalblue", "Model", 2, 1)

    g = data["vaccines_avail_per_hundred"]
    fig = add_line(fig, g, "royalblue", "Model", 2, 2)

    colors = px.colors.qualitative.Pastel

    df = country_data["people_fully_vaccinated_per_hundred"]
    for i, country in enumerate(selected_countries):
        g = df[country].dropna()
        fig = add_line(fig, g, colors[i], country, 1, 1)

    df = country_data["daily_vaccinations_per_million"]
    for i, country in enumerate(selected_countries):
        g = df[country].dropna()
        fig = add_line(fig, g, colors[i], country, 1, 2)

    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_layout(height=700, width=900)

    return fig


@app.callback(
    Output(component_id="output-p-yes-value", component_property="children"),
    Input(component_id="slider-p-yes", component_property="value"),
)
def update_output_div(input_value):
    return r"Completely favorable: {0:.0f}%".format(input_value)


@app.callback(
    Output(component_id="output-p-hard-no-value", component_property="children"),
    Input(component_id="slider-p-hard-no", component_property="value"),
)
def update_output_div(input_value):
    return r"Absolutely against: {0:.0f}%".format(input_value)


@app.callback(
    Output(component_id="output-p-soft-no-value", component_property="children"),
    Input("slider-p-yes", "value"),
    Input("slider-p-hard-no", "value"),
)
def update_output_div(p_yes, p_hard_no):
    v = 100 - (p_yes + p_hard_no)
    return r"Agnosticts: {0:.0f}%".format(v)


@app.callback(
    Output(component_id="output-pressure-value", component_property="children"),
    Input(component_id="slider-pressure", component_property="value"),
)
def update_output_div(input_value):
    return r"Pressure on the agnostics to get vaccinated: {0:.0f}%".format(input_value)


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(host='0.0.0.0', port=80)
