import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import State, Input, Output
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime as dt


from model import run_model_sampling, sample_param_combinations


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


tab_style = {"padding-top": 7}

tab_selected_style = {"padding-top": 7}

tabs_styles = {"height": "44px"}

style_controls = {"height": "33vh"}


def generate_population_controls():

    return html.Div(
        id="population-controls",
        children=[
            html.Div(id="output-p-yes-value"),
            dcc.RangeSlider(
                id="slider-p-yes",
                min=0.0,
                max=100,
                value=[60, 70],
                allowCross=False,
                marks={"0": "0%", "100": "100%"},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-p-yes", storage_type="memory"),
            html.Br(),
            html.Div(id="output-p-hard-no-value"),
            dcc.RangeSlider(
                id="slider-p-hard-no",
                min=0.0,
                max=100,
                value=[15, 25],
                allowCross=False,
                marks={"0": "0%", "100": "100%"},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-p-hard-no", storage_type="memory"),
            html.Br(),
            html.Div(id="agnostics-pct-msg"),
            html.Br(),
            html.Div(id="output-p-soft-no-value"),
            html.Div(id="output-pressure-value"),
            dcc.RangeSlider(
                id="slider-pressure",
                min=0.0,
                max=10,
                value=[2, 5],
                allowCross=False,
                marks={"0": "0%", "10": "10%"},
                step=0.5,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-pressure", storage_type="memory"),
        ],
        style=style_controls,
    )


def generate_vaccine_controls():

    return html.Div(
        id="vaccine-controls",
        children=[
            html.Div(id="output-nv0-value"),
            dcc.RangeSlider(
                id="slider-nv0",
                min=0.0,
                max=1.0,
                value=[0.05, 0.2],
                allowCross=False,
                marks={"0": "0%", "1": "1%"},
                step=0.05,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-nv0", storage_type="memory"),
            html.Br(),
            html.Div(id="output-tau-value"),
            dcc.RangeSlider(
                id="slider-tau",
                min=1,
                max=12,
                value=[4, 5],
                allowCross=False,
                marks={"1": "1 week", "12": "12 weeks"},
                step=0.5,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-tau", storage_type="memory"),
            html.Br(),
            html.Div(id="output-nvmax-value"),
            dcc.RangeSlider(
                id="slider-nvmax",
                min=0.1,
                max=20,
                value=[4, 7],
                allowCross=False,
                marks={"0.1": "0.1%", "20": "20%"},
                step=1e-1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            dcc.Store(id="store-nvmax", storage_type="memory"),
        ],
        style=style_controls,
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
                min=10,
                max=3000,
                step=10,
                debounce=True,
            ),
            dcc.Store(id="store-nrep", storage_type="memory"),
            html.Br(),
            html.Br(),
            html.Div(id="output-N-value"),
            dcc.Input(
                id="input-N",
                type="number",
                value=1000,
                min=300,
                max=10000,
                step=10,
                debounce=True,
            ),
            dcc.Store(id="store-N", storage_type="memory"),
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
            dcc.Store(id="store-CI", storage_type="memory"),
        ],
        style=style_controls,
    )


def generate_country_and_date_controls():

    return html.Div(
        id="contry-date-controls",
        children=[
            html.Hr(style={"height": "2px"}),
            html.P("Date range"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=dt(2020, 12, 30),
                end_date=dt.today(),
                display_format="MMM Do, YY",
                initial_visible_month=dt.today(),
            ),
            dcc.Store(id="store-date-range", storage_type="memory"),
            html.Br(),
            html.Br(),
            html.P("Add country"),
            dcc.Dropdown(
                id="country-select",
                options=[{"label": i, "value": i} for i in avail_countries],
                value=["Germany", "United States", "Russia"],
                multi=True,
            ),
            dcc.Store(id="store-countries", storage_type="memory"),
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
                    # "width": "130vh",
                    # "height": "50vh",
                    # "display": "inline-block",
                    # "overflow": "hidden",
                    # "position": "absolute",
                    # "top": "50%",
                    # "left": "50%",
                    # "transform": "translate(-50%, -50%)"
                },
            ),
        ],
    )


def generate_model_explanation():

    s1 = """                        
        The goal of the model is to capture the **main characteristics** of the **evolution** of an ongoing **vaccination campaign** on a specific population. To this end, the population is described as a sample of discrete **random variables** whose values change according to some **evolution rules**. The model is sampled using the **Monte Carlo method**, i.e., generating random numbers, which are used to simulate the evolution of the random variables over time.
        """
    s2 = """                        
        The **population** is segmented into **three groups**, depending on their views on vaccines:
        
        -   **Pro-vaccines**: they take the vaccine as soon as they have the chance
        -   **Anti-vaccines**: they will never take a vaccine
        -   **Agnostic**: they will initially hesitate, but given enough social pressure, they will take it
        """
    s3 = """ 
        The **evolution of the vaccination** campaign is simulated by applying the following **rules** iteratively where **one iteration** corresponds to **one day**:
        
        1.  Every **non-vaccinated person** in the pro-vaccines group for whom a vaccine is available **becomes vaccinated**. A vaccine becomes available with a probability given by the number of vaccines in stock divided by the population size. That probability is multiplied by 2/7 to account for vaccinations occurring only two days a week, giving an effective per-day probability.
        2.  Every **agnostic person** might **become pro-vaccines** with a probability equal to the number of vaccinated people divided by the population size. This probability is multiplied by a factor, denoted as pressure, which allows for tuning the strength of this effect. This mechanism is a proxy for **social pressure**, i.e., the higher the fraction of vaccinated people is, the higher the influence on non-vaccinated ones to do the same.
        3.  The **stock of vaccines decreases** according to the number of people vaccinated during the day. Care is taken that, each day, no more vaccines than the available stock can be applied.
        """
    s4 = """                        
        The **stock of vaccines is increased** once a week. We distinguish two stages:
        
        1.  Initially, the number of vaccines added to the stock each week **grows exponentially**, representing a fast production growth to meet the demand.
        2.  When a specific **maximum delivery capacity** is reached, that amount does not grow anymore. Every subsequent week, that amount of vaccines are added to the existing stock.
        """
    return html.Div(
        id="text-explanation",
        children=[
            html.Br(),
            dcc.Markdown(s1),
            html.Br(),
            dcc.Markdown(s2),
            html.Br(),
            dcc.Markdown(s3),
            html.Br(),
            dcc.Markdown(s4),
        ],
    )


def generate_model_help():
    return


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H3(
                    "Monte Carlo model of COVID-19 vaccination",
                    style={"color": "#2c8cff"},
                ),
            ],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[
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
                html.Center(html.Button("Run", id="run-button")),
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
                                dcc.Loading(
                                    id="ls-loading-2",
                                    children=[
                                        html.Div([html.Div(id="ls-loading-output-2")])
                                    ],
                                    type="circle",
                                ),
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
                            label="Help",
                            children=[generate_model_help()],
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


def add_line(
    fig, x, y, color, name=None, row=1, col=1, fill="none", width=2, annotation=False
):

    # plot line

    data = dict(
        x=x,
        y=y,
        mode="lines",
        fill=fill,
        line_shape="spline",
        showlegend=False,
        line=dict(color=color, width=width),
    )

    if name is not None:
        data["legendgroup"] = name
        data["name"] = name

    fig.add_trace(
        go.Scatter(data),
        row=row,
        col=col,
    )

    # write annotation
    if annotation:

        fig.add_annotation(
            xref="paper",
            x=x[-1],
            y=y[-1],
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
    # update graph and spinner
    Output("plot_grid", "figure"),
    Output("ls-loading-output-2", "children"),
    # population message: error and agnostic percentage
    Output("agnostics-pct-msg", "children"),
    Output("error-msg", "children"),
    # run button
    Input("run-button", "n_clicks"),
    # countries and dates
    Input("country-select", "value"),
    # population parameters
    State("store-p-yes", "data"),
    State("store-p-hard-no", "data"),
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
)
def update_figures(
    n_clicks,
    # countries
    selected_countries,
    # populatio parameters
    p_yes_bounds,
    p_hard_no_bounds,
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
):

    # default output messages
    msg_agnostics_pct = "Agnosticts: "
    msg_error = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "People vaccinated per hundred",
            "Daily vaccinations per million",
        ),
    )

    colors = px.colors.qualitative.Safe

    if n_clicks is not None:

        # ---- sample the model with the selected parameters ----

        start_date = dt.strptime(date_range["start_date"].split("T")[0], "%Y-%m-%d")
        end_date = dt.strptime(date_range["end_date"].split("T")[0], "%Y-%m-%d")

        # # sliders use values 0-100
        params_combinations, p_soft_no_values = sample_param_combinations(
            np.array(p_yes_bounds) / 100,
            np.array(p_hard_no_bounds) / 100,
            np.array(pressure_bounds) / 100,
            np.array(tau_bounds),
            np.array(nv_0_bounds) / 100,
            np.array(nv_max_bounds) / 100,
            n_rep,
        )

        # evaluate the agnostics population from the pro and anti vaccines samples
        p_soft_no_values = 100 * np.array(p_soft_no_values)
        a = max(np.mean(p_soft_no_values) - np.std(p_soft_no_values), 0)
        b = np.mean(p_soft_no_values) + np.std(p_soft_no_values)
        a_str = "{0:.0f}".format(a)
        b_str = "{0:.0f}".format(b)
        # if the uncertainty interval is smaller than 1%, report one value instead of the interval
        if abs(a - b) < 1:
            msg_agnostics_pct += a_str + "%"
        else:
            msg_agnostics_pct += a_str + " - " + b_str + "%"

        if params_combinations is None:
            msg_error += "ERROR: The pertentages of pro- and anti-vaccines are simultaneously too high. Please reduce them."
            return fig, None, msg_agnostics_pct, msg_error

        data = run_model_sampling(
            params_combinations, start_date, end_date, CI / 100, N
        )

        if data is None:
            msg_error = "ERROR: Maximum computation time of 30s exceeded. Please reduce the number of Monte Carlo runs or the population size."
            return fig, None, msg_agnostics_pct, msg_error

        # ---- plot model results ----

        fig = add_line(
            fig,
            data["pv_dates"],
            data["pv_mean"],
            "royalblue",
            "Model",
            1,
            1,
            annotation=True,
        )
        fig = add_line(
            fig,
            data["pv_dates"],
            data["pv_upper"],
            colors[0],
            f"Model CI={CI}%",
            1,
            1,
            width=0,
            annotation=False,
        )
        fig = add_line(
            fig,
            data["pv_dates"],
            data["pv_lower"],
            colors[0],
            f"Model CI={CI}%",
            1,
            1,
            width=0,
            fill="tonexty",
            annotation=False,
        )

        fig = add_line(
            fig,
            data["dv_dates"],
            data["dv_mean"],
            "royalblue",
            "Model",
            1,
            2,
            annotation=True,
        )
        fig = add_line(
            fig,
            data["dv_dates"],
            data["dv_upper"],
            colors[0],
            f"Model CI={CI}%",
            1,
            2,
            width=0,
            annotation=False,
        )

        data["dv_lower"][data["dv_lower"] < 0] = 0.0
        fig = add_line(
            fig,
            data["dv_dates"],
            data["dv_lower"],
            colors[0],
            f"Model CI={CI}%",
            1,
            2,
            width=0,
            fill="tonexty",
            annotation=False,
        )

    # ----- add curves with data from the selected countries ----

    df = country_data["people_fully_vaccinated_per_hundred"]
    for i, country in enumerate(selected_countries):
        g = df[country].dropna()
        fig = add_line(
            fig, g.index, g, colors[i + 1], country, 1, 1, width=1, annotation=True
        )

    df = country_data["daily_vaccinations_per_million"]
    for i, country in enumerate(selected_countries):
        g = df[country].dropna()
        fig = add_line(
            fig, g.index, g, colors[i + 1], country, 1, 2, width=1, annotation=True
        )

    fig.update_yaxes(range=[0, 100], row=1, col=1)
    # fig.update_layout(height=400, width=1100)

    return fig, None, msg_agnostics_pct, msg_error


@app.callback(
    Output("store-p-yes", "data"),
    Output(component_id="output-p-yes-value", component_property="children"),
    Input(component_id="slider-p-yes", component_property="value"),
    State("store-p-yes", "data"),
)
def update_output_div(values, data):
    data = values
    return data, f"Pro-vaccines: {values[0]} - {values[1]}%"


@app.callback(
    Output("store-p-hard-no", "data"),
    Output(component_id="output-p-hard-no-value", component_property="children"),
    Input(component_id="slider-p-hard-no", component_property="value"),
    State("store-p-hard-no", "data"),
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
    return data, f"Pressure on the agnostics: {values[0]} - {values[1]}%"


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
